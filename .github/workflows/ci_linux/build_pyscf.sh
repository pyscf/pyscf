#!/usr/bin/env bash

set -e -o pipefail

cd ./pyscf/lib
cmake_args=(-DXCFUN_MAX_ORDER=4)
if [ "$(uname -m)" = "x86_64" ]; then
    curl --fail --location "https://media.githubusercontent.com/media/pyscf/pyscf-build-deps/master/pyscf-2.14.1-deps.tar.gz" | tar xzf -
    cmake_args+=(-DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF)
fi
mkdir build; cd build
cmake "${cmake_args[@]}" ..
make -j4
cd ..
rm -Rf build
cd ../..
