#!/usr/bin/env bash

set -e

cd ./pyscf/lib
curl -L "https://github.com/pyscf/pyscf-build-deps/blob/master/pyscf-2.1a-deps.tar.gz?raw=true" | tar xzf -
rm deps/include/cint*
rm deps/lib/libcint*
mkdir build; cd build
#cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF ..
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DCUSTOM_CINT_GIT=https://github.com/fishjojo/libcint.git -DCUSTOM_CINT_GIT_TAG=dev ..
make -j4
cd ..
rm -Rf build
cd ../..
