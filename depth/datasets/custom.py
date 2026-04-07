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
                 data_root,
                 test_mode=True,
                 min_depth=1e-3,
                 max_depth=10,
                 depth_scale=1,
                 img_dir='rgb',
                 depth_dir='depth'):

        self.pipeline = Compose(pipeline)
        self.img_path = os.path.join(data_root, img_dir)
        self.depth_path = os.path.join(data_root, depth_dir)
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        # load annotations
        self.img_infos = self.load_annotations(self.img_path, self.depth_path)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, depth_dir):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory. Load all the images under the root.
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

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

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for img_info in self.img_infos:
            if 'ann' not in img_info:
                raise ValueError(
                    'Ground-truth depth is unavailable for this dataset split.')
            depth_map = osp.join(self.depth_path, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map),
                                      dtype=np.float32) / self.depth_scale
            yield np.expand_dims(depth_map_gt, axis=0)

    def pre_eval(self, preds, indices):
        """Collect evaluation results from each iteration."""
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for pred, index in zip(preds, indices):
            img_info = self.img_infos[index]
            if 'ann' not in img_info:
                raise ValueError(
                    f'Ground-truth depth for index {index} is unavailable.')

            depth_map = osp.join(self.depth_path, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map),
                                      dtype=np.float32) / self.depth_scale
            depth_map_gt = np.expand_dims(depth_map_gt, axis=0)

            eval_result = metrics(
                depth_map_gt,
                pred,
                min_depth=self.min_depth,
                max_depth=self.max_depth)
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
