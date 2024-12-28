set -e
ENV_NAME="unirgb-ir"
source activate $ENV_NAME || conda activate $ENV_NAME

CUDA_HOME=/usr/local/cuda-11.4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
tools/dist_train.sh \
configs/_vpt_cascade-rcnn/FLIR_RGBT_ViTDet/100ep/backbone_vitb_IN1k_mae_coco_224x224/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs.py 8 \
--work-dir work_dirs/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs