# by tsuipo, 23.10.29
import mmcv
from mmcv.transforms import LoadImageFromFile, BaseTransform
import mmengine.fileio as fileio
from mmseg.registry import TRANSFORMS
from typing import Dict, Optional, Union
import numpy as np


@TRANSFORMS.register_module()
class LoadDualModalStackedFromFile(LoadImageFromFile):
    """Load RGB and IR image, then stack them through channel dim``.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # load RGB and IR from `vis_path`, `ir_path`
        vis_path = results['vis_path']
        ir_path = results['ir_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, vis_path)
                vis_bytes = file_client.get(vis_path)
            else:
                vis_bytes = fileio.get(
                    vis_path, backend_args=self.backend_args)
            vis = mmcv.imfrombytes(
                vis_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert vis is not None, f'failed to load image: {vis_path}'
        if self.to_float32:
            vis = vis.astype(np.float32)

        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, ir_path)
                ir_bytes = file_client.get(ir_path)
            else:
                ir_bytes = fileio.get(
                    ir_path, backend_args=self.backend_args)
            ir = mmcv.imfrombytes(
                ir_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert ir is not None, f'failed to load image: {ir_path}'
        if self.to_float32:
            ir = ir.astype(np.float32)
        
        results['img'] = np.concatenate((vis,ir), axis=2)  # NOTE: 6-channel now, images in numpy: H, W, C
        # print('img shape: ', results['img'].shape)
        results['img_shape'] = vis.shape[:2]
        results['ori_shape'] = vis.shape[:2]
        return results
    

@TRANSFORMS.register_module() 
class LoadSODGroundTruth(LoadImageFromFile):
    
    def transform(self, results: dict) -> Optional[dict]:
        gt_path = results['gt_path']
        
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, gt_path)
                gt_bytes = file_client.get(gt_path)
            else:
                gt_bytes = fileio.get(
                    gt_path, backend_args=self.backend_args)
            gt = mmcv.imfrombytes(
                gt_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
            # raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert gt is not None, f'failed to load image: {gt_path}'
        
        gt = gt.astype(np.float32)
            
        gt = gt.mean(axis=2) / 255  # (H, W, 3) -> (H, W)
       
        results['gt'] = gt   
        # print('gt.type: ', gt.dtype)
        
        return results
