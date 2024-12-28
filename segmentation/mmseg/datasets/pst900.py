# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PST900RGBADataset(BaseSegDataset):
    """PST900 dataset.
    image size: [720, 1280]
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # 0:bg, 1:fire-ext, 2:backpack, 3:drill, 4:survivor
        # classes=('fire-ext', 'backpack', 'drill', 'survivor'),
        classes=('bg', 'fire-ext', 'backpack', 'drill', 'survivor'),
        palette=[[255, 255, 255], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]
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
