from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook

from depth.utils.local_visualization import repo_root, save_visualization_triplet


@HOOKS.register_module()
class LocalVisualizationHook(LoggerHook):

    def __init__(self,
                 out_dir='viz',
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(LocalVisualizationHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch)
        self.out_dir = out_dir
        self.run_dir = None

    @staticmethod
    def _get_image_value(log_images, target_key):
        if target_key in log_images:
            return log_images[target_key]
        for key, value in log_images.items():
            if key.endswith(f'.{target_key}') or key.endswith(target_key):
                return value
        return None

    @staticmethod
    def _get_depth_range(runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        decode_head = getattr(model, 'decode_head', None)
        if decode_head is None:
            return None, None
        return getattr(decode_head, 'min_depth', None), getattr(decode_head, 'max_depth', None)

    @master_only
    def before_run(self, runner):
        super(LocalVisualizationHook, self).before_run(runner)
        base_dir = repo_root() / self.out_dir
        if not base_dir.is_absolute():
            base_dir = repo_root() / base_dir
        timestamp = runner.timestamp or 'run'
        self.run_dir = base_dir / timestamp / 'train'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        runner.logger.info(f'Local training visualization will be saved to {self.run_dir}')

    @master_only
    def log(self, runner):
        if self.get_mode(runner) != 'train':
            return

        log_images = runner.outputs.get('log_imgs')
        if not log_images:
            return

        epoch_id = runner.epoch + 1
        iter_id = self.get_iter(runner) + 1
        step_dir = self.run_dir / f'epoch_{epoch_id:04d}_iter_{iter_id:06d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        depth_vmin, depth_vmax = self._get_depth_range(runner)

        save_visualization_triplet(
            step_dir,
            prefix='train',
            img_rgb=self._get_image_value(log_images, 'img_rgb'),
            depth_pred=self._get_image_value(log_images, 'img_depth_pred'),
            depth_gt=self._get_image_value(log_images, 'img_depth_gt'),
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax)
