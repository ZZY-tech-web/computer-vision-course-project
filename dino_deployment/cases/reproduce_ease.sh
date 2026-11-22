#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 EASE_DIR SD_IMAGE_DIR TEST_DATASET_DIR [HQ_SAM_CHECKPOINT]"
  exit 2
fi

ease_dir=$1
sd_image_dir=$2
test_dataset_dir=$3
hq_sam_checkpoint=${4:-}

for required_file in \
  gen-envir-proto.py \
  ret-unsupervised.py \
  ret-prompt-based-seg.py \
  envir-cate.yaml; do
  if [[ ! -f "${ease_dir}/${required_file}" ]]; then
    echo "Missing EASE file: ${ease_dir}/${required_file}" >&2
    exit 1
  fi
done

if [[ ! -d "${sd_image_dir}" ]]; then
  echo "Stable Diffusion image directory not found: ${sd_image_dir}" >&2
  exit 1
fi

if [[ ! -d "${test_dataset_dir}" ]]; then
  echo "Test dataset directory not found: ${test_dataset_dir}" >&2
  exit 1
fi

ease_dir=$(realpath "${ease_dir}")
sd_image_dir=$(realpath "${sd_image_dir}")
test_dataset_dir=$(realpath "${test_dataset_dir}")

cd "${ease_dir}"

python gen-envir-proto.py \
  --category_path envir-cate.yaml \
  --sd_img_path "${sd_image_dir}" \
  --proto_save_path ./prototype \
  --img_size 476 \
  --dino vitl \
  --device cuda

python ret-unsupervised.py \
  --proto_path ./prototype \
  --test_dataset_path "${test_dataset_dir}" \
  --mask_save_path ./pred/unsupervised \
  --img_size 476 \
  --dino vitl \
  --device cuda \
  --faiss_device cuda

if [[ -n "${hq_sam_checkpoint}" ]]; then
  hq_sam_checkpoint=$(realpath "${hq_sam_checkpoint}")
  python ret-prompt-based-seg.py \
    --proto_path ./prototype \
    --test_dataset_path "${test_dataset_dir}" \
    --mask_save_path ./pred/prompt-based-seg \
    --sam_ckpt_path "${hq_sam_checkpoint}" \
    --img_size 476 \
    --dino vitl \
    --device cuda \
    --faiss_device cuda
fi
