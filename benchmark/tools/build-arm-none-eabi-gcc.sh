#!/bin/bash -e

#Create a directory for installation
export TARGET=arm-none-eabi
export OUTPUT_DIR=`pwd`/${TARGET}-gcc
mkdir -p ${OUTPUT_DIR}
export PATH=${OUTPUT_DIR}/bin:$PATH

#Toolchain version
export BINUTIL_VER=2.23
export GCC_VER=4.7.2
export NEWLIB_VER=1.20.0

#Install dependencies

apt-get install build-essential g++ texinfo \
                     libgmp3-dev libmpc-dev libmpfr-dev

#Download the Source Tarball
wget ftp://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-${GCC_VER}/gcc-${GCC_VER}.tar.bz2
wget http://ftp.gnu.org/gnu/binutils/binutils-${BINUTIL_VER}.tar.gz
wget ftp://sources.redhat.com/pub/newlib/newlib-${NEWLIB_VER}.tar.gz

#Build binutils

tar zxvf binutils-${BINUTIL_VER}.tar.gz
mkdir binutils-${BINUTIL_VER}-build
cd binutils-${BINUTIL_VER}-build
../binutils-${BINUTIL_VER}/configure \
  --prefix=${OUTPUT_DIR} \
  --target=${TARGET}
make -j6
make install
cd ..

#Copy newlib include directory

tar zxvf newlib-${NEWLIB_VER}.tar.gz

#Build gcc

tar jxvf gcc-${GCC_VER}.tar.bz2
mkdir gcc-${GCC_VER}-build
cd gcc-${GCC_VER}-build
../gcc-${GCC_VER}/configure \
  --prefix=${OUTPUT_DIR} \
  --enable-static \
  --disable-shared \
  --enable-languages=c,c++ \
  --disable-nls \
  --disable-bootstrap \
  --with-newlib \
  --with-headers="../newlib-${NEWLIB_VER}/newlib/libc/include" \
  --target=${TARGET}
make -j6
make install
cd ..

#Build newlib
mkdir newlib-${NEWLIB_VER}-build
cd newlib-${NEWLIB_VER}-build
../newlib-${NEWLIB_VER}/configure \
  --prefix=${OUTPUT_DIR} \
  --target=${TARGET} \
  --enable-static \
  --disable-shared
make -j6
make install
cd ..

#Pack tarball
tar jcvf ${OUTPUT_DIR}.tar.bz2 ${OUTPUT_DIR}
