#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

BUILD_DIR="${ROOT}/build"
SIMCONFIG_DIR="${ROOT}/simconfig"
BENCHMARK_DIR="${ROOT}/benchmark"

TASKS="${BENCHMARK_DIR}/matrix_simple.ini"

for cfg in ${SIMCONFIG_DIR}/*; do
  echo "### Running ${cfg} ..."
  ${BUILD_DIR}/TileSim "${cfg}" ${TASKS}
done