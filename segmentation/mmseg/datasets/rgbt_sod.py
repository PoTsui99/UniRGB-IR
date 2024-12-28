# By tsuipo, 2023.10.27
# For RGBT salient object detection task
# Follow the original origanizing format of VT5000, VT1000, VT821, VT723
# Recommended to follow the classic play of Using VT5000 train/ to train
# Then test on VT1000, VT821, VT723
import os.path as osp
from typing import List

import mmengine.fileio as fileio
from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
# from .my_basesegdataset import MyBaseSegDataset
from mmengine.dataset import BaseDataset, Compose
from typing import Optional, Sequence, Union, Callable
import copy

@DATASETS.register_module()
class VTDataset(BaseDataset):
    """
        Root
        ├── VT5000
        │   ├── Train(current directory)
        │   │    ├── RGB
        │   │    │   ├── 0.jpg
        │   │    │   │── ...        
        │   │    │── T
        │   │    │   ├── 0.jpg
        │   │    │   │── ...   
        │   │    │── GT
        │   │    │   ├── 0.png
        │   │    │   │── ...  
        │   │    │   
        │   ├── Test
        │   │    ├── RGB
        │   │    │   ├── 0.jpg
        │   │    │   │── ...        
        │   │    │── T
        │   │    │   ├── 0.jpg
        │   │    │   │── ...   
        │   │    │── GT
        │   │    │   ├── 0.png
        │   │    │   │── ...         
    """
    
    # METAINFO: dict = dict()
   
    def __init__(self,
                 data_prefix=dict(
                     vis_path='Train/RGB', ir_path='Train/T', gt_path='Train/GT'),
                 vis_suffix='.jpg',
                 ir_suffix='.jpg',
                 gt_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.ann_file = None
        self.data_root = data_root
        self.vis_suffix = vis_suffix
        self.ir_suffix = ir_suffix
        self.gt_suffix = gt_suffix
        
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        # self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        vis_dir = self.data_prefix.get('vis_path', None) 
        ir_dir = self.data_prefix.get('ir_path', None) 
        gt_dir = self.data_prefix.get('gt_path', None)

        _suffix_len = len(self.vis_suffix)
        for img in fileio.list_dir_or_file(  # img: name of rgb image(full name)
                dir_path=vis_dir,
                list_dir=False,
                suffix=self.vis_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(vis_path=osp.join(vis_dir, img))  # 'img_path'
            # import pdb; pdb.set_trace()
            ir_path = img[:-_suffix_len] + self.ir_suffix
            data_info['ir_path'] = osp.join(ir_dir, ir_path)
            if gt_dir is not None:  # load annotation
                gt_path = img[:-_suffix_len] + self.gt_suffix  # fname + suffix(.png)
                data_info['gt_path'] = osp.join(gt_dir, gt_path)  # 'gt_path'
    
            data_list.append(data_info)
            
        data_list = sorted(data_list, key=lambda x: x['vis_path'])
        
        return data_list  # return a list of dictionary


# @DATASETS.register_module()
# class VTDataset(MyBaseSegDataset):
#     """
#         Root
#         ├── VT5000
#         │   ├── Train(current directory)
#         │   │    ├── RGB
#         │   │    │   ├── 0.jpg
#         │   │    │   │── ...        
#         │   │    │── T
#         │   │    │   ├── 0.jpg
#         │   │    │   │── ...   
#         │   │    │── GT
#         │   │    │   ├── 0.png
#         │   │    │   │── ...  
#         │   │    │   
#         │   ├── Test
#         │   │    ├── RGB
#         │   │    │   ├── 0.jpg
#         │   │    │   │── ...        
#         │   │    │── T
#         │   │    │   ├── 0.jpg
#         │   │    │   │── ...   
#         │   │    │── GT
#         │   │    │   ├── 0.png
#         │   │    │   │── ...         
#     """
    
#     METAINFO = None
   
#     def __init__(self,
#                 data_prefix=dict(
#                     vis_path='Train/RGB', ir_path='Train/T', gt_path='Train/GT'),
#                 vis_suffix='.jpg',
#                 ir_suffix='.jpg',
#                 gt_suffix='.png',
#                 **kwargs) -> None:
#         super().__init__(  # HACK: dont use this to set attribute for CLS
#             data_prefix=data_prefix,
#             vis_suffix=vis_suffix,
#             ir_suffix=ir_suffix,
#             gt_suffix=gt_suffix,
#             **kwargs)
#         self.data_prefix = data_prefix  # directory 
#         self.vis_suffix = vis_suffix
#         self.ir_suffix = ir_suffix
#         self.gt_suffix = gt_suffix
        
        
    # def load_data_list(self) -> List[dict]:
    #     """Load annotation from directory or annotation file.

    #     Returns:
    #         list[dict]: All data info of dataset.
    #     """
    #     data_list = []
    #     vis_dir = self.data_prefix.get('vis_path', None) 
    #     ir_dir = self.data_prefix.get('ir_path', None) 
    #     gt_dir = self.data_prefix.get('gt_path', None)

    #     _suffix_len = len(self.vis_suffix[0])
    #     for img in fileio.list_dir_or_file(  # get file of given suffix...
    #             dir_path=vis_dir,
    #             list_dir=False,
    #             suffix=self.vis_suffix,
    #             recursive=True,
    #             backend_args=self.backend_args):
    #         data_info = dict(vis_path=osp.join(vis_dir, img))  # 'img_path'
    #         ir_path = img[:-_suffix_len] + self.ir_suffix[0]
    #         data_info['ir_path'] = osp.join(ir_dir, ir_path)
    #         if gt_dir is not None:  # load annotation
    #             gt_path = img[:-_suffix_len] + self.gt_suffix  # fname + suffix(.png)
    #             data_info['gt_path'] = osp.join(gt_dir, gt_path)
    #         # data_info['seg_fields'] = []
    #         # data_info['category_id'] = self._get_category_id_from_filename(img)
    #         data_list.append(data_info)
            
    #     data_list = sorted(data_list, key=lambda x: x['vis_path'])
        
    #     return data_list  # return a list of dictionary
    