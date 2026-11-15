#!/usr/bin/env bash

set -e -o pipefail

cd ./pyscf/lib

cmake_args=(-DXCFUN_MAX_ORDER=4)
if [ "$(uname -m)" = "x86_64" ]; then
    # Download to a file so retries can discard partial transfers.
    curl --fail --location --retry 3 --output deps.tar.gz \
        "https://media.githubusercontent.com/media/pyscf/pyscf-build-deps/refs/heads/master/pyscf-2.14.1-deps.tar.gz"
    tar xzf deps.tar.gz
    rm -f deps.tar.gz
    cmake_args+=(-DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF)
fi

mkdir build; cd build
cmake "${cmake_args[@]}" ..
make -j4
cd ..
rm -Rf build
cd ../..
