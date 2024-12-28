# Adapted from OpenMMLab. by tsuipo, 24.4.6
# this detector is for visualization.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor
from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from pathlib import Path
import numpy as np
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
class TwoStreamTwoStageDetectorVisualizer(BaseDetector):  
    """Visualizer.
    for feature map & t-SNE
    """

    def __init__(self,
                 backbone: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
    
        # collate the output feature and middleway feature
        feat_dict = self.backbone(batch_inputs['img'], batch_inputs['img_ir']) 
        
        return feat_dict

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        feat_dict = self.extract_feat(batch_inputs)
        x = feat_dict['output_feat']

        if self.with_neck:
            x = self.neck(x)
            
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
 
        feat_dict = self.extract_feat(batch_inputs)
        x = feat_dict['output_feat']
        
        if self.with_neck:
            x = self.neck(x)
            
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)  # add the loss factors from RPN module
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)  # add the loss factors from ROI module

        return losses  # return loss

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        feat_dict = self.extract_feat(batch_inputs)
        x = feat_dict['output_feat']
        
        if self.with_neck:
            x = self.neck(x)
        
        save_root = Path.cwd() / 'work_dirs' / 'vis_32tsne_8featmap'
        # if not save_root.exists():
        #     save_root.mkdir(parents=True)
        
        # visualize the batch samples
        for i, sample in enumerate(batch_data_samples):  # NOTE: for each sample in this batch
            fname = sample.img_path.split('/')[-1][:-8]  # e.g. FLIR_08864
            save_dir = save_root / fname
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            
            ori_vis_fname = sample.img_path
            ori_ir_fname = sample.img_ir_path
            ori_vis = cv2.imread(ori_vis_fname)
            ori_ir = cv2.imread(ori_ir_fname)
            cv2.imwrite(str(save_dir / 'ori_vis.png'), ori_vis)
            cv2.imwrite(str(save_dir / 'ori_ir.png'), ori_ir)
            
            # vit_input: RGB feat, cnn_input_16x: IR feat
            vit_input = feat_dict['vit_input_stage3'][i, ...]  # seq_len, dim
            cnn_input_viz = feat_dict['cnn_input_32x'][i, ...]
            cnn_input_tsne = feat_dict['cnn_input_8x'][i, ...]
            prompt = feat_dict['prompt_stage3'][i, ...]
            # extracted_feat_stage0 = feat_dict['extracted_feat_stage0'][i, ...]
            
            vit_input = vit_input.transpose(0, 1).reshape(768, 64, 64)  # C, H, W, 768*64*64
            cnn_input_tsne = cnn_input_tsne.transpose(0, 1).reshape(768, 128, 128) # C, H, W
            cnn_input_viz = cnn_input_viz.transpose(0, 1).reshape(768, 32, 32) # C, H, W
            prompt = prompt.transpose(0, 1).reshape(768, 64, 64)  # C, H, W
            
            # resize to 16x downsample
            size = (64, 64)
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
            
            
            vis_all_channel = False
            img_size = sample.ori_shape
            alpha = 0.5  # hyperparameter
            
            # if vis_all_channel:
            #     for i in range(768):
            #         vit_input = torch.tensor(vit_input[i]).unsqueeze(2)  # ->H, W, 1
            #         cnn_input_viz = torch.tensor(cnn_input_viz[i]).unsqueeze(2)
            #         prompt = torch.tensor(prompt[i]).unsqueeze(2)
            #         vit_input = cv2.applyColorMap(feature_vis_upsample(vit_input, img_size), 2)
            #         cnn_input_viz = cv2.applyColorMap(feature_vis_upsample(cnn_input_viz, img_size), 2)
            #         prompt = cv2.applyColorMap(feature_vis_upsample(prompt, img_size), 2)
                    
            #         overlay_vit_input = vit_input * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
            #         overlay_cnn_input = cnn_input_viz * alpha + ori_ir * (1 - alpha)         # overlay to ir img
            #         overlay_prompt = prompt * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
                    
            #         cv2.imwrite(str(save_dir / f'vit_feat_ch{i}.png'), overlay_vit_input)
            #         cv2.imwrite(str(save_dir / f'cnn_feat_ch{i}.png'), overlay_cnn_input)
            #         cv2.imwrite(str(save_dir / f'inject_feat_ch{i}.png'), overlay_prompt)
            # else:
            #     # to single-channel image
            #     vit_input = vit_input.mean(dim=0).unsqueeze(2)  # ->H, W, 1
            #     cnn_input_viz = cnn_input_viz.mean(dim=0).unsqueeze(2)  
            #     prompt = prompt.mean(dim=0).unsqueeze(2)  
            #     # extracted_feat_stage0 = extracted_feat_stage0.mean(dim=0).unsqueeze(2) 
                
                
            #     # dye the featmap
            #     vit_input = cv2.applyColorMap(feature_vis_upsample(vit_input, img_size), 2)
            #     cnn_input_viz = cv2.applyColorMap(feature_vis_upsample(cnn_input_viz, img_size), 2)
            #     prompt = cv2.applyColorMap(feature_vis_upsample(prompt, img_size), 2)
            #     # extracted_feat_stage0 = cv2.applyColorMap(feature_vis_upsample(extracted_feat_stage0, img_size), 2)
                
            #     overlay_vit_input = vit_input * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
            #     overlay_cnn_input = cnn_input_viz * alpha + ori_ir * (1 - alpha)         # overlay to ir img
            #     overlay_prompt = prompt * alpha + ori_ir * (1 - alpha)  # overlay to rgb img
            #     # overlay_extracted_feat_stage0 = extracted_feat_stage0 * alpha + ori_vis * (1 - alpha)  # overlay to rgb img
                
            #     cv2.imwrite(str(save_dir / 'vit_feat.png'), overlay_vit_input)
            #     cv2.imwrite(str(save_dir / 'cnn_feat.png'), overlay_cnn_input)
            #     cv2.imwrite(str(save_dir / 'inject_feat.png'), overlay_prompt)
            #     # cv2.imwrite(str(save_dir / 'inject_feat.png'), overlay_extracted_feat_stage0)
            
                
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:  # use the rpn network to generate proposal
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
