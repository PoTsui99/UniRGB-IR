set -e
ENV_NAME="unirgb-ir"
source activate $ENV_NAME || conda activate $ENV_NAME

python tools/train.py configs/_vpt_cascade-rcnn/FLIR_RGBT_ViTDet/100ep/backbone_vitb_IN1k_mae_coco_224x224/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs.py --work-dir work_dirs/1024_v15_rgb-ir-feat-fusion-spm_crossAttn-GRU_staged_8x1bs
