#!/usr/bin/env bash

set -e

cd ./pyscf/lib
curl -L "https://github.com/pyscf/pyscf-build-deps/blob/master/pyscf-2.8a-deps.tar.gz?raw=true" | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF -DXCFUN_MAX_ORDER=4 ..
make -j4
cd ..
rm -Rf build
cd ../..
