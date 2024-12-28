set -e
ENV_NAME="unirgb-ir"
source activate $ENV_NAME || conda activate $ENV_NAME

CUDA_HOME=/usr/local/cuda-11.4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
tools/dist_train.sh \
configs/_vpt_rgbt_salient/SETR/ssim_416_iou_mla_v15_8x8b_20k.py \
--work-dir work_dirs/sod_ssim_416_iou_mla_v15_8x8b_20k
