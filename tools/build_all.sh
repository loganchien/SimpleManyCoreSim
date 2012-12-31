#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

BUILD_DIR="${ROOT}/build"

mkdir -p "${BUILD_DIR}" || true
cd "${BUILD_DIR}"
cmake "${ROOT}"
make -j6
