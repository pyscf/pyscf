#!/usr/bin/env bash

set -e

cd ./pyscf/lib
curl -L "https://github.com/pyscf/pyscf-build-deps/blob/master/pyscf-2.2.1-deps.tar.gz?raw=true" | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF ..
make -j4
cd ..
rm -Rf build
cd ../..
