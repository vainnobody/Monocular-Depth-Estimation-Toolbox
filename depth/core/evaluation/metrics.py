from collections import OrderedDict

import mmcv
import numpy as np
import torch

STANDARD_DEPTH_METRIC_NAMES = (
    'a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog',
    'sq_rel')


def _mask_valid_depth(gt, pred, min_depth=1e-3, max_depth=80):
    mask = np.ones_like(gt, dtype=bool)
    if min_depth is not None:
        mask = np.logical_and(mask, gt > min_depth)
    if max_depth is not None:
        mask = np.logical_and(mask, gt < max_depth)

    gt = gt[mask]
    pred = pred[mask]
    return gt, pred


def _standard_metrics_dict(gt, pred):
    if gt.shape[0] == 0:
        return OrderedDict((name, np.nan) for name in STANDARD_DEPTH_METRIC_NAMES)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)

    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return OrderedDict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel)


def _reference_height(values, mode='median'):
    if values.shape[0] == 0:
        return np.nan
    if isinstance(mode, str) and mode.startswith('p'):
        percentile = float(mode[1:])
        return np.percentile(values, percentile)
    if mode == 'median':
        return np.median(values)
    if mode == 'mean':
        return np.mean(values)
    if mode == 'min':
        return np.min(values)
    raise ValueError(f'Unsupported relative height reference mode: {mode}')


def calculate(gt, pred):
    ret = _standard_metrics_dict(gt, pred)
    return tuple(ret[name] for name in STANDARD_DEPTH_METRIC_NAMES)


def metrics(gt, pred, min_depth=1e-3, max_depth=80, as_dict=False):
    gt, pred = _mask_valid_depth(gt, pred, min_depth=min_depth, max_depth=max_depth)
    ret = _standard_metrics_dict(gt, pred)
    if as_dict:
        return ret
    return tuple(ret[name] for name in STANDARD_DEPTH_METRIC_NAMES)


def eval_metrics(gt, pred, min_depth=1e-3, max_depth=80):
    gt, pred = _mask_valid_depth(gt, pred, min_depth=min_depth, max_depth=max_depth)
    return dict(_standard_metrics_dict(gt, pred))


def relative_height_metrics(gt,
                            pred,
                            min_depth=1e-3,
                            max_depth=80,
                            reference='median'):
    gt, pred = _mask_valid_depth(gt, pred, min_depth=min_depth, max_depth=max_depth)

    if gt.shape[0] == 0:
        return OrderedDict(rel_bias=np.nan, rel_mae=np.nan, rel_rmse=np.nan)

    gt_rel = gt - _reference_height(gt, reference)
    pred_rel = pred - _reference_height(pred, reference)
    diff = pred_rel - gt_rel

    return OrderedDict(
        rel_bias=np.mean(diff),
        rel_mae=np.mean(np.abs(diff)),
        rel_rmse=np.sqrt(np.mean(diff ** 2)))


def pre_eval_to_metrics(pre_eval_results):
    if len(pre_eval_results) == 0:
        return {}

    if isinstance(pre_eval_results[0], dict):
        ret_metrics = OrderedDict()
        metric_names = list(pre_eval_results[0].keys())
        for metric_name in metric_names:
            ret_metrics[metric_name] = np.nanmean(
                [result.get(metric_name, np.nan) for result in pre_eval_results])
        return dict(ret_metrics)

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
    ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
    ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
    ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
    ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
    ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
    ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
    ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
    ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }

    return ret_metrics
