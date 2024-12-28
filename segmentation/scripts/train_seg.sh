set -e
ENV_NAME="unirgb-ir"
source activate $ENV_NAME || conda activate $ENV_NAME

CUDA_HOME=/usr/local/cuda-11.4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
tools/dist_train.sh \
configs/_vpt_rgbt_segmentation/MFNet/SETR/mla_v15_mae_coco_2xb2_80k_withBG.py \
--work-dir work_dirs/mfnet_mla_v15_mae_coco_2xb2_80k_withBG
