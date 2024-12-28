# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from pathlib import Path
import numpy as np
import torch
# for draw the diagrams
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn import manifold
import cv2
import torch.nn.functional as F


def UMAP_visualize(feat_vis, feat_ir, fname, label1='vis feat', label2='ir feat'):
    sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=1.2, color_codes=True, rc=None)
    """ input shape: H, W, C
    """
    feat_vis = feat_vis.cpu()
    feat_ir = feat_ir.cpu()
    plt.figure(figsize=(8, 8))
    
    # flatten to: seq_len, C(feature dim)
    feat_vis_flat = torch.flatten(feat_vis, start_dim=0, end_dim=1)
    feat_ir_flat = torch.flatten(feat_ir, start_dim=0, end_dim=1)
    
    # Initialize UMAP
    umap = UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=921)
    
    # Fit and transform with UMAP
    vis_umap = umap.fit_transform(feat_vis_flat)
    ir_umap = umap.fit_transform(feat_ir_flat)
    
    # Normalize for better visualization
    x_min, x_max = vis_umap.min(0), vis_umap.max(0)
    vis_norm = (vis_umap - x_min) / (x_max - x_min)
    
    x_min, x_max = ir_umap.min(0), ir_umap.max(0)
    ir_norm = (ir_umap - x_min) / (x_max - x_min)
    
    # Plotting
    plt.scatter(vis_norm[:, 0], vis_norm[:, 1], c='green', label=label1, alpha=0.9, s=1)
    plt.scatter(ir_norm[:, 0], ir_norm[:, 1], c='red', label=label2, alpha=0.9, s=1)
    
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def feature_vis_upsample(feature, size):
    x_min, x_max = feature.min(), feature.max()
    X_norm = (feature - x_min) / (x_max - x_min)  # 归一化
    # X_norm = 1 - X_norm
    X_norm = torch.clamp((X_norm - (X_norm.min() + 0.3)) / (X_norm.max() - (X_norm.min() + 0.3)), 0, 1)

    X_norm_up = F.interpolate(X_norm.permute(2, 0, 1).unsqueeze(0), size=size, mode='bilinear').squeeze(0).permute(1, 2, 0)

    feature = np.array(X_norm_up.cpu())
    return np.uint8(255 * feature)


def tSNE_visualize(feat_vis, feat_ir, fname, label1='vis feat', label2='ir feat', perplexity=50, early_exaggeration=12):
    sns.set(context='paper', style='white', palette='deep', font='sans-serif', font_scale=1.2, color_codes=True, rc=None)
    """ input shape: H, W, C
    """
    feat_vis = feat_vis.cpu()
    feat_ir = feat_ir.cpu()
    plt.figure(figsize=(8, 8))
    
    # flatten to: seq_len, C(feature dim)
    feat_vis_flat = torch.flatten(feat_vis, start_dim=0, end_dim=1)
    feat_ir_flat = torch.flatten(feat_ir, start_dim=0, end_dim=1)
    
    tsne = manifold.TSNE(n_components=2, init='pca', 
                         perplexity=perplexity, early_exaggeration=early_exaggeration, random_state=921)
    # tsne = manifold.TSNE(n_components=2, init='pca', perplexity=80, random_state=921)
    # tsne = manifold.TSNE(n_components=2, random_state=42)
    
    vis_tsne = tsne.fit_transform(feat_vis_flat)
    x_min, x_max = vis_tsne.min(0), vis_tsne.max(0)
    x_norm = (vis_tsne - x_min) / (x_max - x_min)
    plt.scatter(x_norm[:, 0], x_norm[:, 1], c='green', label=label1, alpha=0.9, s=1)
    
    ir_tsne = tsne.fit_transform(feat_ir_flat)
    x_min, x_max = ir_tsne.min(0), ir_tsne.max(0)
    x_norm = (ir_tsne - x_min) / (x_max - x_min)
    plt.scatter(x_norm[:, 0], x_norm[:, 1], c='red', label=label2, alpha=0.9, s=1)
    # # 将两个特征图合并为一个大的数据集进行t-SNE降维
    # combined_features = torch.cat((feat_vis_flat, feat_ir_flat), dim=0)
    
    # 应用t-SNE降维
    # combined_tsne_results = tsne.fit_transform(combined_features)
    
    # 分割降维后的结果以便可视化
    # num_vis_samples = feat_vis_flat.shape[0]
    # vis_tsne = combined_tsne_results[:num_vis_samples, :]
    # ir_tsne = combined_tsne_results[num_vis_samples:, :]
    
    # # 可视化
    # # plt.figure(figsize=(8, 8))
    # plt.scatter(vis_tsne[:, 0], vis_tsne[:, 1], c='green', label='feat_vis', alpha=0.5)
    # plt.scatter(ir_tsne[:, 0], ir_tsne[:, 1], c='red', label='feat_ir', alpha=0.5)
    plt.legend()
    # plt.title("t-SNE Visualization of feat_vis (Green) and feat_ir (Red)")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close() 


@MODELS.register_module()
class EncoderDecoderVisualizer(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor, data_samples) -> List[Tensor]:
        """Extract features from images."""
        feat_dict = self.backbone(inputs)
        x = feat_dict['outs']
        if self.with_neck:
            x = self.neck(x)
            
        save_root = Path.cwd() / 'work_dirs' / 'vis_32tsne_8featmap'
        
        # visualize the batch samples
        for i, sample in enumerate(data_samples):  # NOTE: for each sample in this batch
            fname = sample.img_path.split('/')[-1][:-4]  # e.g. 58_bag21_rect_rgb_frame000000
            save_dir = save_root / fname
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            
            ori_rgba_fname = sample.img_path
            bgra = cv2.imread(ori_rgba_fname, -1)
            b, g, r, a = cv2.split(bgra)
            ori_vis = cv2.merge([b, g, r])  # H, W, 3
            ori_ir = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(str(save_dir / 'ori_vis.png'), ori_vis)
            cv2.imwrite(str(save_dir / 'ori_ir.png'), ori_ir)
            
            # vit_input: RGB feat, cnn_input_16x: IR feat
            vit_input = feat_dict['vit_input_stage3'][i, ...]  # seq_len, dim
            cnn_input_viz = feat_dict['cnn_input_32x'][i, ...]
            cnn_input_tsne = feat_dict['cnn_input_8x'][i, ...]
            prompt = feat_dict['prompt_stage3'][i, ...]
            # extracted_feat_stage0 = feat_dict['extracted_feat_stage0'][i, ...]
            
            vit_input = vit_input.transpose(0, 1).reshape(768, 48, 48)  # C, H, W, 768*48*48
            cnn_input_tsne = cnn_input_tsne.transpose(0, 1).reshape(768, 96, 96) # C, H, W
            cnn_input_viz = cnn_input_viz.transpose(0, 1).reshape(768, 24, 24) # C, H, W
            prompt = prompt.transpose(0, 1).reshape(768, 48, 48)  # C, H, W
            
            # resize to 16x downsample
            size = (48, 48)
            cnn_input_viz = F.interpolate(cnn_input_viz.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
            cnn_input_tsne = F.interpolate(cnn_input_tsne.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
            # extracted_feat_stage0 = extracted_feat_stage0.transpose(0, 1).reshape(768, 64, 64)  # C, H, W
            
            # t-SNE visualization
            tSNE_visualize(vit_input.permute(1, 2, 0), cnn_input_tsne.permute(1, 2, 0),
                           str(save_dir / 'tSNE_vit-cnn_diverge.png'), label1=r'$F_v$', label2=r'$F_{mfp}$',
                           perplexity=80, early_exaggeration=8)
            # UMAP_visualize(vit_input_stage0.permute(1, 2, 0), cnn_input_16x.permute(1, 2, 0),
                        #    str(save_dir / 'UMAP_vit-cnn_diverge.png'), label1=r'$F_v$', label2=r'$F_{mfp}$')
            tSNE_visualize(vit_input.permute(1, 2, 0), prompt.permute(1, 2, 0),
                           str(save_dir / 'tSNE_vit-prompt_converge.png'), label1=r'$F_v$', label2=r'$F_s$',
                           perplexity=80, early_exaggeration=8)
            # UMAP_visualize(vit_input_stage0.permute(1, 2, 0), prompt_stage0.permute(1, 2, 0),
                        #    str(save_dir / 'UMAP_vit-prompt_converge.png'), label1=r'$F_v$', label2=r'$F_{mfp}$')
            
            
            # to single-channel image
            vit_input = vit_input.mean(dim=0).unsqueeze(2)  # H, W, C
            cnn_input_viz = cnn_input_viz.mean(dim=0).unsqueeze(2)  # H, W, C
            prompt = prompt.mean(dim=0).unsqueeze(2)  # H, W, C
            # extracted_feat_stage0 = extracted_feat_stage0.mean(dim=0).unsqueeze(2)  # H, W, C
            
            
            # dye the featmap
            img_size = sample.ori_shape
            vit_input = cv2.applyColorMap(feature_vis_upsample(vit_input, img_size), 2)
            cnn_input_viz = cv2.applyColorMap(feature_vis_upsample(cnn_input_viz, img_size), 2)
            prompt = cv2.applyColorMap(feature_vis_upsample(prompt, img_size), 2)
            # extracted_feat_stage0 = cv2.applyColorMap(feature_vis_upsample(extracted_feat_stage0, img_size), 2)
            
            alpha = 0.5  # hyperparameter
            overlay_vit_input = vit_input * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
            overlay_cnn_input = cnn_input_viz * alpha + ori_ir * (1 - alpha)         # overlay to ir img
            overlay_prompt = prompt * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
            # overlay_extracted_feat_stage0 = extracted_feat_stage0 * alpha + ori_vis * (1 - alpha)  # overlay to rgb img
            
            cv2.imwrite(str(save_dir / 'vit_feat.png'), overlay_vit_input)
            cv2.imwrite(str(save_dir / 'cnn_feat.png'), overlay_cnn_input)
            cv2.imwrite(str(save_dir / 'inject_feat.png'), overlay_prompt)
            # cv2.imwrite(str(save_dir / 'inject_feat.png'), overlay_extracted_feat_stage0)
        
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict],
                      batch_data_samples) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs, batch_data_samples)
        
        
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # NOTE: call the loss() of decode head
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs, data_samples)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas, data_samples)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas, batch_data_samples)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict],
                        data_samples) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas, data_samples)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict], data_samples) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas, data_samples)  # XXX: visualizer only supports whole_inference

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
