#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

BUILD_DIR="${ROOT}/build"
SIMCONFIG_DIR="${ROOT}/simconfig"
BENCHMARK_DIR="${ROOT}/benchmark"

TASKS="${BENCHMARK_DIR}/matrix_simple_16.ini"

#TASKS="
#${BENCHMARK_DIR}/helloworld.ini
#${BENCHMARK_DIR}/matrix_simple_16.ini
#${BENCHMARK_DIR}/matrix_simple_256.ini
#"

"${SCRIPT_DIR}/build_all.sh" "$1"

ulimit -t 3600
ulimit -v 1048576
ulimit -a

i=0
for task in ${TASKS}; do
  for cfg in ${SIMCONFIG_DIR}/*; do
    echo "### Running ${cfg} ..."
    if [ "$1" = "debug" ]; then
      gdb --args ${BUILD_DIR}/TileSim "${cfg}" "${task}"
    else
      time ${BUILD_DIR}/TileSim "${cfg}" "${task}" 2>&1 | tee "run$i.log"
    fi
  done

  i=$(echo "$i + 1" | bc)
done
