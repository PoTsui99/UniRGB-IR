# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MFNetRGBADataset(BaseSegDataset):
    """MFNet dataset.
    image size: [480, 640]
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump
        # as ade.py, label 0 stands for background, is ignored. 
        classes=('unlabeled', 'car', 'person', 
                 'bike', 'curve', 'car_stop', 
                 'guardrail', 'color_cone', 'bump'),
        palette=[[255, 255, 255], [128, 64, 128], [244, 35, 232], 
                 [70, 70, 70], [102, 102, 156], [190, 153, 153], 
                 [153, 153, 153], [250, 170, 30], [220, 220, 0]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
