import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from PIL import Image

import os


@DATASETS.register_module()
class CustomDepthDataset(Dataset):
    """Custom dataset for supervised monocular depth esitmation. 
    An example of file structure. is as followed.
    .. code-block:: none
        ├── data
        │   ├── custom
        │   │   ├── train
        │   │   │   ├── rgb
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   │   ├── depth
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   ├── val
        │   │   │   ...
        │   │   │   ...

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        data_root (str, optional): Data root for img_dir.
        test_mode (bool): test_mode=True
        min_depth=1e-3: Default min depth value.
        max_depth=10: Default max depth value.
    """

    def __init__(self,
                 pipeline,
                 data_root=None,
                 test_mode=True,
                 min_depth=1e-3,
                 max_depth=10,
                 depth_scale=1,
                 img_dir='rgb',
                 depth_dir='depth',
                 split=None,
                 eval_min_depth=None,
                 eval_max_depth=None):

        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_path = self._resolve_path(data_root, img_dir)
        self.depth_path = self._resolve_path(data_root, depth_dir)
        self.split = self._resolve_split_path(data_root, split)
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.eval_min_depth = min_depth if eval_min_depth is None else eval_min_depth
        self.eval_max_depth = max_depth if eval_max_depth is None else eval_max_depth
        self.depth_scale = depth_scale

        # load annotations
        self.img_infos = self.load_annotations(self.img_path, self.depth_path,
                                               self.split)
        self._pre_eval_diagnostics = []
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def _resolve_path(self, data_root, path):
        if path is None:
            return None
        if data_root is not None and not osp.isabs(path):
            return osp.join(data_root, path)
        return path

    def _resolve_split_path(self, data_root, split):
        if split is None:
            return None
        if osp.isabs(split):
            return split
        candidate = osp.join(data_root, split) if data_root is not None else split
        if osp.exists(candidate):
            return candidate
        return split

    def load_annotations(self, img_dir, depth_dir, split=None):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory. Load all the images under the root.
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        if split is not None:
            with open(split) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    fields = line.split()
                    if len(fields) == 1:
                        img_name, depth_name = fields[0], fields[0]
                    else:
                        img_name, depth_name = fields[:2]

                    img_info = dict(filename=img_name)
                    if depth_name != 'None':
                        img_info['ann'] = dict(depth_map=depth_name)
                    elif not self.test_mode:
                        raise FileNotFoundError(
                            f'Cannot find matched depth map for image "{img_name}" in split "{split}".')
                    img_infos.append(img_info)
        else:
            imgs = sorted(os.listdir(img_dir))
            depths = sorted(os.listdir(depth_dir)) if osp.isdir(depth_dir) else []
            depth_lookup = {depth_name: depth_name for depth_name in depths}

            for img in imgs:
                img_info = dict()
                img_info['filename'] = img
                if img in depth_lookup:
                    img_info['ann'] = dict(depth_map=depth_lookup[img])
                elif not self.test_mode:
                    raise FileNotFoundError(
                        f'Cannot find matched depth map for image "{img}" in "{depth_dir}".')
                img_infos.append(img_info)

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())

        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.img_path
        results['depth_prefix'] = self.depth_path
        results['depth_scale'] = self.depth_scale

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if 'ann' in img_info:
            results['ann_info'] = img_info['ann']
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']
    
    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0] * self.depth_scale) # Do not convert to np.uint16 for ensembling. # .astype(np.uint16)
        return results

    def load_gt_depth_map(self, idx, expand_dim=True):
        img_info = self.img_infos[idx]
        if 'ann' not in img_info:
            raise ValueError(f'Ground-truth depth for index {idx} is unavailable.')

        depth_map = osp.join(self.depth_path, img_info['ann']['depth_map'])
        depth_map_gt = np.asarray(Image.open(depth_map),
                                  dtype=np.float32) / self.depth_scale
        if expand_dim:
            depth_map_gt = np.expand_dims(depth_map_gt, axis=0)
        return depth_map_gt

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for idx, _ in enumerate(self.img_infos):
            yield self.load_gt_depth_map(idx, expand_dim=True)

    def _collect_pre_eval_diagnostics(self, depth_map_gt, pred):
        depth_map_gt = np.asarray(depth_map_gt, dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != depth_map_gt.shape:
            pred = np.reshape(pred, depth_map_gt.shape)

        valid_mask = np.logical_and(depth_map_gt > self.eval_min_depth,
                                    depth_map_gt < self.eval_max_depth)
        total_pixels = int(valid_mask.size)
        valid_pixels = int(valid_mask.sum())
        valid_ratio = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

        diagnostic = dict(
            valid_ratio=valid_ratio,
            valid_pixels=valid_pixels,
            total_pixels=total_pixels,
        )

        if valid_pixels == 0:
            diagnostic.update(
                gt_min=np.nan,
                gt_p1=np.nan,
                gt_p99=np.nan,
                gt_max=np.nan,
                pred_min=np.nan,
                pred_p1=np.nan,
                pred_p99=np.nan,
                pred_max=np.nan)
            return diagnostic

        gt_valid = depth_map_gt[valid_mask]
        pred_valid = pred[valid_mask]
        gt_percentiles = np.percentile(gt_valid, [1.0, 99.0])
        pred_percentiles = np.percentile(pred_valid, [1.0, 99.0])
        diagnostic.update(
            gt_min=float(gt_valid.min()),
            gt_p1=float(gt_percentiles[0]),
            gt_p99=float(gt_percentiles[1]),
            gt_max=float(gt_valid.max()),
            pred_min=float(pred_valid.min()),
            pred_p1=float(pred_percentiles[0]),
            pred_p99=float(pred_percentiles[1]),
            pred_max=float(pred_valid.max()))
        return diagnostic

    def _summarize_pre_eval_diagnostics(self, logger=None):
        if not self._pre_eval_diagnostics:
            return

        diagnostics = self._pre_eval_diagnostics
        valid_ratios = np.array([item['valid_ratio'] for item in diagnostics], dtype=np.float32)
        valid_pixels = np.array([item['valid_pixels'] for item in diagnostics], dtype=np.float32)
        total_pixels = np.array([item['total_pixels'] for item in diagnostics], dtype=np.float32)

        def _summary(key):
            values = np.array([item[key] for item in diagnostics], dtype=np.float32)
            return float(np.nanmean(values))

        summary_table_data = PrettyTable()
        summary_table_data.add_column('diag', [
            'valid_ratio_mean',
            'valid_ratio_min',
            'valid_pixels_mean',
            'total_pixels_mean',
            'gt_min_mean',
            'gt_p1_mean',
            'gt_p99_mean',
            'gt_max_mean',
            'pred_min_mean',
            'pred_p1_mean',
            'pred_p99_mean',
            'pred_max_mean',
        ])
        summary_table_data.add_column('value', [
            np.round(float(np.nanmean(valid_ratios)), 6),
            np.round(float(np.nanmin(valid_ratios)), 6),
            np.round(float(np.nanmean(valid_pixels)), 2),
            np.round(float(np.nanmean(total_pixels)), 2),
            np.round(_summary('gt_min'), 4),
            np.round(_summary('gt_p1'), 4),
            np.round(_summary('gt_p99'), 4),
            np.round(_summary('gt_max'), 4),
            np.round(_summary('pred_min'), 4),
            np.round(_summary('pred_p1'), 4),
            np.round(_summary('pred_p99'), 4),
            np.round(_summary('pred_max'), 4),
        ])

        print_log('Pre-eval diagnostics:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)
        self._pre_eval_diagnostics = []

    def pre_eval(self, preds, indices):
        """Collect evaluation results from each iteration."""
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for pred, index in zip(preds, indices):
            depth_map_gt = self.load_gt_depth_map(index, expand_dim=True)
            self._pre_eval_diagnostics.append(
                self._collect_pre_eval_diagnostics(depth_map_gt, pred))

            eval_result = metrics(
                depth_map_gt,
                pred,
                min_depth=self.eval_min_depth,
                max_depth=self.eval_max_depth)
            pre_eval_results.append(eval_result)
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset."""
        metric = [
            "a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log",
            "silog", "sq_rel"
        ]
        eval_results = {}

        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(gt_depth_maps, results)
        else:
            ret_metrics = pre_eval_to_metrics(results)
            self._summarize_pre_eval_diagnostics(logger=logger)

        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i * 9:i * 9 + 9]
            values = ret_metric_values[i * 9:i * 9 + 9]

            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
