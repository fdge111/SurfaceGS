export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
gpu_id=0

SOURCE_PATH=XXX
MODEL_PATH=XXX

CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
--config configs/waymo/reconstruction/017.yaml \
source_path=$SOURCE_PATH \
model_path=$MODEL_PATH
