
set -e
ONNX_PATH=${1:-artifacts/model.onnx} # path of onnx model
ENGINE_PATH=${2:-artifacts/model.plan} # path of tensorrt model
PRECISION=${3:-fp16} # fp16 | fp32 | int8
WORKSPACE=${4:-4096} # MiB

command -v trtexec >/dev/null 2>&1 || {
  echo >&2 "trtexec not found."
  exit 1
}

echo "Building TensorRT engine"
precision_flag="--${PRECISION}"

trtexec  --onnx="${ONNX_PATH}" \
         --saveEngine="${ENGINE_PATH}" \
         ${precision_flag} \
         --noTF32 \
         --minShapes=input_ids:1x512,attention_mask:1x512 \
         --optShapes=input_ids:4x512,attention_mask:4x512 \
         --maxShapes=input_ids:8x512,attention_mask:8x512 \
         --workspace=${WORKSPACE} \
         --verbose

echo "TensorRT engine saved → ${ENGINE_PATH}"
