set -e
ENV_NAME="unirgb-ir"
source activate $ENV_NAME || conda activate $ENV_NAME

CUDA_HOME=/usr/local/cuda-11.4 CUDA_VISIBLE_DEVICES=0,1 \
tools/dist_train.sh \
configs/_vpt_ablation/flir_768_v15_ir-vit_ir-rgb-adapter_4stage_2x2bs.py \
--work-dir work_dirs/ablation/768_v15_ir-vit_ir-rgb-adapter_4stage_2x2bs