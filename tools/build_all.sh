#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

BUILD_DIR="${ROOT}/build"

mkdir -p "${BUILD_DIR}" || true
cd "${BUILD_DIR}"

if [ "$1" = "debug" ]; then
  echo "Build the debug build ..."
  cmake -DCMAKE_BUILD_TYPE=Debug "${ROOT}"
else
  echo "Build the release build ..."
  cmake "${ROOT}"
fi

make -j6
