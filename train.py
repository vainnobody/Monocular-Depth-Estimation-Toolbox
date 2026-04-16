# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmcv_custom import *
from mmrotate.utils import register_all_modules
import torch
from torch import nn
from collections import OrderedDict
from collections import defaultdict
from mmengine.dist import is_main_process
from typing import Dict, Tuple, List, Optional

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # parser.add_argument(
    #     '--load-mapped',
    #     help='load a custom checkpoint by mapping keys (encoder->backbone, rotdetdecoder->neck/rpn/roi)',
    #     default=None
    # )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args




def load_rot_decoder_correctly(
    model,
    ckpt_path: str,
    decoder_branch: str = "auto",   # "auto" | "opt" | "sar" | "both" | "none"
    verbose: bool = True,
):
    """Load backbone + detection decoder weights from a custom checkpoint into mmrotate model.

    Mapping rules:
      - encoder.* -> backbone.*
      - rotdetdecoder[.<branch>].neck.*     -> neck.*
      - rotdetdecoder[.<branch>].rpn_head.* -> rpn_head.*
      - rotdetdecoder[.<branch>].roi_head.* -> roi_head.*

    Where <branch> can be: opt / sar.
    - decoder_branch="auto": prefer opt if exists, else sar if exists, else fallback to legacy (no branch).
    - decoder_branch="opt"/"sar": load only that branch if exists; if not exists -> do nothing for decoder.
    - decoder_branch="both": try load both branches sequentially (later one may overwrite same keys; usually不建议).
    - decoder_branch="none": do not load any decoder, only load backbone.
    """

    assert decoder_branch in {"auto", "opt", "sar", "both", "none"}

    real_model = model.module if hasattr(model, "module") else model
    model_state = real_model.state_dict()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # -------- detect available decoder branches in ckpt --------
    has_opt = any(k.startswith("module.rotdetdecoder.opt.") or k.startswith("rotdetdecoder.opt.")
                  for k in ckpt_state.keys())
    has_sar = any(k.startswith("module.rotdetdecoder.sar.") or k.startswith("rotdetdecoder.sar.")
                  for k in ckpt_state.keys())
    has_legacy = any(k.startswith("module.rotdetdecoder.neck.") or k.startswith("rotdetdecoder.neck.")
                     for k in ckpt_state.keys())

    def choose_branches() -> List[Optional[str]]:
        """Return list of branches to load: [None] means legacy no-branch format."""
        if decoder_branch == "none":
            return []

        if decoder_branch == "both":
            branches = []
            if has_opt:
                branches.append("opt")
            if has_sar:
                branches.append("sar")
            if not branches and has_legacy:
                branches.append(None)
            return branches

        if decoder_branch == "auto":
            if has_opt:
                return ["opt"]
            if has_sar:
                return ["sar"]
            if has_legacy:
                return [None]
            return []  # no decoder keys at all

        # decoder_branch is "opt" or "sar"
        if decoder_branch == "opt":
            return ["opt"] if has_opt else ([] if not has_legacy else [None])
        if decoder_branch == "sar":
            return ["sar"] if has_sar else ([] if not has_legacy else [None])
        return []

    branches_to_load = choose_branches()

    # -------- mapping + filtering --------
    mapped: Dict[str, torch.Tensor] = {}
    skipped_nomatch: List[Tuple[str, str]] = []
    skipped_shape: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = []

    def add_if_match(src_key: str, dst_key: str, tensor: torch.Tensor):
        if dst_key not in model_state:
            skipped_nomatch.append((src_key, dst_key))
            return
        if model_state[dst_key].shape != tensor.shape:
            skipped_shape.append((src_key, dst_key, tuple(tensor.shape), tuple(model_state[dst_key].shape)))
            return
        mapped[dst_key] = tensor

    # 1) always load backbone (encoder -> backbone)
    for k, v in ckpt_state.items():
        k2 = k[7:] if k.startswith("module.") else k
        if k2.startswith("encoder."):
            dst = "backbone." + k2[len("encoder."):]
            add_if_match(k2, dst, v)

    # 2) load decoder by selected branch(es)
    for br in branches_to_load:
        # legacy no branch
        if br is None:
            prefix = "rotdetdecoder."
        else:
            prefix = f"rotdetdecoder.{br}."

        for k, v in ckpt_state.items():
            k2 = k[7:] if k.startswith("module.") else k
            if not k2.startswith(prefix):
                continue

            # map to mmdet keys: remove "rotdetdecoder." and optional "<branch>."
            if br is None:
                # rotdetdecoder.neck.xxx -> neck.xxx
                if k2.startswith("rotdetdecoder.neck.") or k2.startswith("rotdetdecoder.rpn_head.") or k2.startswith("rotdetdecoder.roi_head."):
                    dst = k2.replace("rotdetdecoder.", "", 1)
                    add_if_match(k2, dst, v)
            else:
                # rotdetdecoder.opt.neck.xxx -> neck.xxx
                # rotdetdecoder.sar.rpn_head.xxx -> rpn_head.xxx
                if (k2.startswith(f"rotdetdecoder.{br}.neck.") or
                    k2.startswith(f"rotdetdecoder.{br}.rpn_head.") or
                    k2.startswith(f"rotdetdecoder.{br}.roi_head.")):
                    dst = k2.replace(f"rotdetdecoder.{br}.", "", 1)
                    add_if_match(k2, dst, v)

    msg = real_model.load_state_dict(mapped, strict=False)

    if verbose and is_main_process():
        print_log(f"[Mapped loader] ckpt: {ckpt_path}", logger='current')
        print_log(f"[Mapped loader] decoder_branch={decoder_branch}, "
                  f"detected: opt={has_opt}, sar={has_sar}, legacy={has_legacy}, "
                  f"using={branches_to_load}", logger='current')
        print_log(f"[Mapped loader] mapped keys: {len(mapped)}", logger='current')
        print_log(f"[Mapped loader] skipped (not in model): {len(skipped_nomatch)}", logger='current')
        print_log(f"[Mapped loader] skipped (shape mismatch): {len(skipped_shape)}", logger='current')
        if skipped_shape:
            print_log("[Mapped loader] examples of shape mismatches:", logger='current')
            for item in skipped_shape[:12]:
                print_log(f"  {item[0]} -> {item[1]}  ckpt{item[2]} vs model{item[3]}", logger='current')
        print_log(f"[Mapped loader] load_state_dict: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}",
                  logger='current')

    return msg





def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # if args.load_mapped is not None:
        # load after runner built, before training
    load_rot_decoder_correctly(runner.model, '/data2/users/yangcong2/sslworkplace/weights/pretrained/zzy/dinov3_vit_b_mae_0.5_sparsemoefc2_lt_k3_t1_semim3p_iter_10k.pth', decoder_branch="opt", verbose=True)

    runner.train()


if __name__ == '__main__':
    main()
