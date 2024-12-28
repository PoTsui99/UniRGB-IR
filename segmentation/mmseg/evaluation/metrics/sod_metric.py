from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from typing import Dict, List, Optional, Sequence
import torch
import numpy as np
from mmengine.dist import is_main_process
from mmengine.utils import mkdir_or_exist
from collections import OrderedDict
from PIL import Image
import os.path as osp
import cv2
# import torch.nn.functional as F

@METRICS.register_module()
class DummyMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results: list) -> Dict[str, float]:
        return dict()

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        pass

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        return dict()

@METRICS.register_module()
class MAEMetric(BaseMetric):
    """Mean Absolute Error evaluation metric for segmentation."""

    def __init__(self, collect_device: str = 'cpu', output_dir: Optional[str] = None, **kwargs) -> None:
        super().__init__(collect_device=collect_device)
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.results = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.
        Compute the absolute difference between predicted labels and ground truth labels.
        """
        for data, data_sample in zip(data_batch['data_samples'], data_samples):
            pred_label = data_sample['seg_logits']['data'].squeeze()
            label = data.gt_sem_seg.data.squeeze().to('cpu')
            
            # to 0-255
            pred_label = (torch.sigmoid(pred_label) * 255).to('cpu')
            label = label * 255  # 0/1 logical map
            # resize to (224, 224)
            pred_label = cv2.resize(pred_label.numpy(), (224, 224), interpolation=cv2.INTER_AREA)
            label = cv2.resize(label.numpy(), (224, 224), interpolation=cv2.INTER_AREA)
            # back to 0-1
            pred_label = torch.from_numpy(pred_label) / 255
            label = (torch.from_numpy(label) > 128).float()
            
            error = torch.abs(pred_label - label)
            self.results.extend(error.flatten().tolist())  # Collect all pixel-wise error(in list)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the mean absolute error from collected results."""
        mae = np.mean(self.results)
        return {'MAE': mae}  # return a dict

    @staticmethod
    def save_output(data_sample, pred_label):
        """Optional: Save output predictions as images."""
        if 'output_dir' in data_sample:
            basename = osp.splitext(osp.basename(data_sample['img_path']))[0]
            png_filename = osp.abspath(osp.join(data_sample['output_dir'], f'{basename}.png'))
            output = Image.fromarray(pred_label.cpu().numpy().astype(np.uint8))
            output.save(png_filename)
