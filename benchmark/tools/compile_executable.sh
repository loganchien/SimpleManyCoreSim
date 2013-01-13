#!/bin/bash -e

OUTPUT_NAME="${1}"
INPUT_FILE="${2}"
THREAD_WIDTH="${3}"
THREAD_HEIGHT="${4}"
BLOCK_WIDTH="${5}"
BLOCK_HEIGHT="${6}"

shift
shift
shift
shift
shift
shift

if [ -z "${OUTPUT_NAME}" ]; then
  echo "ERROR: OUTPUT_NAME is not available."
  exit 1
fi

if [ -z "${INPUT_FILE}" ]; then
  echo "ERROR: INPUT_FILE is not available."
  exit 1
fi

if [ -z "${THREAD_WIDTH}" ]; then
  echo "ERROR: THREAD_WIDTH is not available."
  exit 1
fi

if [ -z "${THREAD_HEIGHT}" ]; then
  echo "ERROR: THREAD_HEIGHT is not available."
  exit 1
fi

if [ -z "${BLOCK_WIDTH}" ]; then
  echo "ERROR: BLOCK_WIDTH is not available."
  exit 1
fi

if [ -z "${BLOCK_HEIGHT}" ]; then
  echo "ERROR: BLOCK_HEIGHT is not available."
  exit 1
fi

OUTPUT_DIR="$(dirname "${INPUT_FILE}")/binaries"
OUTPUT_EXECUTABLE="${OUTPUT_DIR}/${OUTPUT_NAME}"
OUTPUT_CONFIG="${OUTPUT_DIR}/${OUTPUT_NAME}.ini"

if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir -p "${OUTPUT_DIR}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
RT_DIR="$(cd "${SCRIPT_DIR}/../rt"; pwd)"


CC="arm-none-eabi-gcc"
CFLAGS="-static -mthumb -Bstatic -O0
        -include ${RT_DIR}/lib.h ${RT_DIR}/lib.c $*"

OBJDUMP="arm-none-eabi-objdump"

if [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: You must specify a source code"
  exit 1
fi

${CC} ${CFLAGS} "${INPUT_FILE}" -o "${OUTPUT_EXECUTABLE}"

rm -f "${OUTPUT_CONFIG}"
echo "[task]" >> "${OUTPUT_CONFIG}"
echo "executable=${OUTPUT_EXECUTABLE}" >> "${OUTPUT_CONFIG}"

get_addr () {
  ${OBJDUMP} -t ${OUTPUT_EXECUTABLE} | grep $1 | awk '{ print $1 }'
}

echo "thread_idx_addr=$(get_addr "threadIdx")" >> "${OUTPUT_CONFIG}"
echo "thread_dim_addr=$(get_addr "threadDim")" >> "${OUTPUT_CONFIG}"
echo "block_idx_addr=$(get_addr "blockIdx")" >> "${OUTPUT_CONFIG}"
echo "block_dim_addr=$(get_addr "blockDim")" >> "${OUTPUT_CONFIG}"
echo "thread_width=${THREAD_WIDTH}" >> "${OUTPUT_CONFIG}"
echo "thread_height=${THREAD_HEIGHT}" >> "${OUTPUT_CONFIG}"
echo "block_width=${BLOCK_WIDTH}" >> "${OUTPUT_CONFIG}"
echo "block_height=${BLOCK_HEIGHT}" >> ${OUTPUT_CONFIG}
