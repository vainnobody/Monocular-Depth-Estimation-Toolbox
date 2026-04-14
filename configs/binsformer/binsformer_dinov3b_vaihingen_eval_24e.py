import os
from pathlib import Path

_base_ = ['./binsformer_dinov3b_vaihingen_24e.py']


def _resolve_train_work_dir():
    for env_key in ('BINSFORMER_VAIHINGEN_WORK_DIR', 'MONODEPTH_WORK_DIR'):
        env_work_dir = os.getenv(env_key)
        if env_work_dir:
            return Path(env_work_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / 'work_dirs' / 'binsformer_dinov3b_vaihingen_24e'


def _resolve_eval_checkpoint(work_dir: Path) -> str:
    search_patterns = (
        'best_abs_rel*.pth',
        'best*.pth',
        'latest.pth',
    )

    for pattern in search_patterns:
        matches = list(work_dir.glob(pattern))
        if not matches:
            continue
        checkpoint = max(matches, key=lambda path: path.stat().st_mtime)
        return str(checkpoint)

    raise FileNotFoundError(
        'No checkpoint found for eval config. Expected one of '
        f'{search_patterns} under {work_dir}. You can override the work dir '
        'with BINSFORMER_VAIHINGEN_WORK_DIR or MONODEPTH_WORK_DIR.')


work_dir = str(_resolve_train_work_dir())
load_from = _resolve_eval_checkpoint(Path(work_dir))
