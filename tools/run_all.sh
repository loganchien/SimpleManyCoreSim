#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

BUILD_DIR="${ROOT}/build"
SIMCONFIG_DIR="${ROOT}/simconfig"
BENCHMARK_DIR="${ROOT}/benchmark/binaries"

TASKS=""

TASKS="${TASKS}
${BENCHMARK_DIR}/helloworld.ini
"

TASKS="${TASKS}
${BENCHMARK_DIR}/matrix_simple_16_1_16.ini
${BENCHMARK_DIR}/matrix_simple_16_2_8.ini
${BENCHMARK_DIR}/matrix_simple_16_4_4.ini
${BENCHMARK_DIR}/matrix_simple_16_8_2.ini
${BENCHMARK_DIR}/matrix_simple_16_16_1.ini
"

TASKS="${TASKS}
${BENCHMARK_DIR}/matrix_simple_32_1_32.ini
${BENCHMARK_DIR}/matrix_simple_32_2_16.ini
${BENCHMARK_DIR}/matrix_simple_32_4_8.ini
${BENCHMARK_DIR}/matrix_simple_32_8_4.ini
${BENCHMARK_DIR}/matrix_simple_32_16_2.ini
${BENCHMARK_DIR}/matrix_simple_32_32_1.ini
"

TASKS="${TASKS}
${BENCHMARK_DIR}/matrix_simple_64_1_64.ini
${BENCHMARK_DIR}/matrix_simple_64_2_32.ini
${BENCHMARK_DIR}/matrix_simple_64_4_16.ini
${BENCHMARK_DIR}/matrix_simple_64_8_8.ini
${BENCHMARK_DIR}/matrix_simple_64_16_4.ini
${BENCHMARK_DIR}/matrix_simple_64_32_2.ini
${BENCHMARK_DIR}/matrix_simple_64_64_1.ini
"

#TASKS="${TASKS}
#${BENCHMARK_DIR}/matrix_simple_128_1_128.ini
#${BENCHMARK_DIR}/matrix_simple_128_2_64.ini
#${BENCHMARK_DIR}/matrix_simple_128_4_32.ini
#${BENCHMARK_DIR}/matrix_simple_128_8_16.ini
#${BENCHMARK_DIR}/matrix_simple_128_16_8.ini
#${BENCHMARK_DIR}/matrix_simple_128_32_4.ini
#${BENCHMARK_DIR}/matrix_simple_128_64_2.ini
#${BENCHMARK_DIR}/matrix_simple_128_128_1.ini
#"

#TASKS="${TASKS}
#${BENCHMARK_DIR}/matrix_simple_256_1_256.ini
#${BENCHMARK_DIR}/matrix_simple_256_2_128.ini
#${BENCHMARK_DIR}/matrix_simple_256_4_64.ini
#${BENCHMARK_DIR}/matrix_simple_256_8_32.ini
#${BENCHMARK_DIR}/matrix_simple_256_16_16.ini
#${BENCHMARK_DIR}/matrix_simple_256_32_8.ini
#${BENCHMARK_DIR}/matrix_simple_256_64_4.ini
#${BENCHMARK_DIR}/matrix_simple_256_128_2.ini
#${BENCHMARK_DIR}/matrix_simple_256_256_1.ini
#"

#TASKS="${TASKS}
#${BENCHMARK_DIR}/matrix_simple_512_1_512.ini
#${BENCHMARK_DIR}/matrix_simple_512_2_256.ini
#${BENCHMARK_DIR}/matrix_simple_512_4_128.ini
#${BENCHMARK_DIR}/matrix_simple_512_8_64.ini
#${BENCHMARK_DIR}/matrix_simple_512_16_32.ini
#${BENCHMARK_DIR}/matrix_simple_512_32_16.ini
#${BENCHMARK_DIR}/matrix_simple_512_64_8.ini
#${BENCHMARK_DIR}/matrix_simple_512_128_4.ini
#${BENCHMARK_DIR}/matrix_simple_512_256_2.ini
#${BENCHMARK_DIR}/matrix_simple_512_512_1.ini
#"

"${SCRIPT_DIR}/build_all.sh" "$1"

ulimit -t 3600
ulimit -v 1048576
ulimit -a

i=0
for cfg in ${SIMCONFIG_DIR}/*; do
  short_cfg=${cfg:${#ROOT}+1}
  echo "=== CONFIG: ${short_cfg}"
  for task in ${TASKS}; do
    short_task=${task:${#ROOT}+1}
    echo "# Task: ${short_task} ..."
    if [ "$1" = "debug" ]; then
      gdb --args ${BUILD_DIR}/TileSim "${cfg}" "${task}"
    else
      ${BUILD_DIR}/TileSim "${cfg}" "${task}"
      if [ "$?" != "0" ]; then
        echo "Finished with error: $?"
      fi
    fi
  done

  i=$(echo "$i + 1" | bc)
done
