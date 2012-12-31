#!/bin/bash

# This script will scan and copy the minimal subset of libboost into the
# directory: external/boost.

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.."; pwd)"

SRC="${ROOT}/src"
BOOST="${ROOT}/external/boost_1_52_0"
BOOST_MIN="${ROOT}/external/boost"
BOOST_FILE="${ROOT}/external/boost_1_52_0.tar.bz2"
BOOST_URL="http://nchc.dl.sourceforge.net/project/boost/boost/1.52.0/boost_1_52_0.tar.bz2"

if [ ! -d "${BOOST}" ]; then
  if [ ! -f "${BOOST_FILE}" ]; then
    echo "### Downloading boost libraries ..."
    rm -rf "${BOOST_FILE}" > /dev/null 2>&1
    wget "${BOOST_URL}" -O "${BOOST_FILE}"
  fi

  echo "### Extracting boost libraries ..."
  rm -rf "${BOOST}" > /dev/null 2>&1
  mkdir -p "${BOOST}"
  tar jxf "${BOOST_FILE}" -C "${BOOST}" --strip=1
fi

echo "### Removing existing directory ..."
rm -rf "${BOOST_MIN}"

echo "### Scan for boost libraries ..."
mkdir -p "${BOOST_MIN}"
find "${SRC}" -type f -name '*.cpp' | \
  xargs -I{} bcp --boost="${BOOST}" --scan {} "${BOOST_MIN}" 2> /dev/null
