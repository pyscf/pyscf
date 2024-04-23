#!/usr/bin/env bash
set -e

#XXX default clang compiler does not support openmp, shall we use gcc?
cd ./pyscf/lib
#curl -L https://github.com/fishjojo/pyscf-deps/raw/master/pyscf-1.7.5-deps-macos-10.14.tar.gz | tar xzf -
mkdir build; cd build
#cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF ..
cmake -DCMAKE_Fortran_COMPILER=/Applications/Xcode_14.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/gfortran ..
make -j4
cd ..
rm -Rf build
cd ../..
