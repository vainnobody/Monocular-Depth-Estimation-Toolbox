# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from pathlib import Path

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm

from depth.utils.local_visualization import repo_root


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
    # greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 pre_eval=False,
                 save_viz=False,
                 viz_dir='viz',
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.save_viz = save_viz
        self.viz_dir = viz_dir

    def _get_viz_dir(self, runner):
        if not self.save_viz:
            return None
        base_dir = Path(self.viz_dir)
        if not base_dir.is_absolute():
            base_dir = repo_root() / base_dir
        epoch_or_iter = runner.epoch + 1 if self.by_epoch else runner.iter + 1
        scope = 'epoch' if self.by_epoch else 'iter'
        return base_dir / (runner.timestamp or 'run') / 'val' / f'{scope}_{epoch_or_iter:04d}'

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from depth.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            pre_eval=self.pre_eval,
            save_viz_dir=self._get_viz_dir(runner))
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
    # greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 pre_eval=False,
                 save_viz=False,
                 viz_dir='viz',
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.save_viz = save_viz
        self.viz_dir = viz_dir

    def _get_viz_dir(self, runner):
        if not self.save_viz:
            return None
        base_dir = Path(self.viz_dir)
        if not base_dir.is_absolute():
            base_dir = repo_root() / base_dir
        epoch_or_iter = runner.epoch + 1 if self.by_epoch else runner.iter + 1
        scope = 'epoch' if self.by_epoch else 'iter'
        return base_dir / (runner.timestamp or 'run') / 'val' / f'{scope}_{epoch_or_iter:04d}'

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from depth.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval,
            save_viz_dir=self._get_viz_dir(runner))

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
