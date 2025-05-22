export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="trained-flux-dev-dreambooth-lora"
export INSTANCE_DIR="/data/sjl/DATA"

/data/sjl/.conda/envs/dif/bin/accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_text_encoder\
  --instance_prompt="achieve a transformation of {}" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="achieve a transformation of {}" \
  --seed="0" \
  --push_to_hub