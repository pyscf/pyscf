#!/usr/bin/env bash

set -e -o pipefail

cd ./pyscf/lib
curl --fail --location "https://media.githubusercontent.com/media/pyscf/pyscf-build-deps/master/pyscf-2.14.1-deps.tar.gz" | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF -DXCFUN_MAX_ORDER=4 ..
make -j4
cd ..
rm -Rf build
cd ../..
