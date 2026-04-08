import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import (eval_metrics, metrics, pre_eval_to_metrics,
                        relative_height_metrics)
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
                 eval_max_depth=None,
                 relative_eval=False,
                 relative_eval_ref='median',
                 standard_metrics_on_relative=False,
                 relative_metric_min_depth=1.0,
                 relative_metric_max_depth=None):

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
        self.relative_eval = relative_eval
        self.relative_eval_ref = relative_eval_ref
        self.standard_metrics_on_relative = standard_metrics_on_relative
        self.relative_metric_min_depth = relative_metric_min_depth
        self.relative_metric_max_depth = relative_metric_max_depth
        self.depth_scale = depth_scale

        # load annotations
        self.img_infos = self.load_annotations(self.img_path, self.depth_path,
                                               self.split)
        

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

    def _to_relative_depth(self, depth_map_gt, pred, eps=1e-3):
        abs_mask = np.ones_like(depth_map_gt, dtype=bool)
        if self.eval_min_depth is not None:
            abs_mask = np.logical_and(abs_mask, depth_map_gt > self.eval_min_depth)
        if self.eval_max_depth is not None:
            abs_mask = np.logical_and(abs_mask, depth_map_gt < self.eval_max_depth)

        valid_gt = depth_map_gt[abs_mask]
        if valid_gt.size == 0:
            return depth_map_gt, pred

        if isinstance(self.relative_eval_ref, str) and self.relative_eval_ref.startswith('p'):
            percentile = float(self.relative_eval_ref[1:])
            base_height = np.percentile(valid_gt, percentile)
        elif self.relative_eval_ref == 'median':
            base_height = np.median(valid_gt)
        elif self.relative_eval_ref == 'mean':
            base_height = np.mean(valid_gt)
        elif self.relative_eval_ref == 'min':
            base_height = np.min(valid_gt)
        else:
            raise ValueError(f'Unsupported relative_eval_ref: {self.relative_eval_ref}')

        depth_map_gt = np.maximum(depth_map_gt - base_height, eps)
        pred = np.maximum(pred - base_height, eps)
        return depth_map_gt, pred

    def _compute_metrics(self, depth_map_gt, pred):
        if isinstance(pred, str):
            pred = np.load(pred)
        metric_gt = depth_map_gt
        metric_pred = pred
        metric_min_depth = self.eval_min_depth
        metric_max_depth = self.eval_max_depth
        if self.standard_metrics_on_relative:
            metric_gt, metric_pred = self._to_relative_depth(depth_map_gt, pred)
            metric_min_depth = self.relative_metric_min_depth
            metric_max_depth = self.relative_metric_max_depth
        eval_result = metrics(
            metric_gt,
            metric_pred,
            min_depth=metric_min_depth,
            max_depth=metric_max_depth,
            as_dict=True)
        if self.relative_eval:
            eval_result.update(
                relative_height_metrics(
                    depth_map_gt,
                    pred,
                    min_depth=self.eval_min_depth,
                    max_depth=self.eval_max_depth,
                    reference=self.relative_eval_ref))
        return eval_result

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
            eval_result = self._compute_metrics(depth_map_gt, pred)
            pre_eval_results.append(eval_result)
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset."""
        eval_results = {}

        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            per_image_metrics = []
            for depth_map_gt, pred in zip(self.get_gt_depth_maps(), results):
                per_image_metrics.append(self._compute_metrics(depth_map_gt, pred))
            ret_metrics = pre_eval_to_metrics(per_image_metrics)
        else:
            ret_metrics = pre_eval_to_metrics(results)

        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        for start in range(0, len(ret_metrics), 9):
            names = ret_metric_names[start:start + 9]
            values = ret_metric_values[start:start + 9]

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
