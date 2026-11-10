#!/usr/bin/env bash

set -e -o pipefail

cd ./pyscf/lib
# Download to a file so retries can discard partial transfers.
curl --fail --location --retry 3 --output deps.tar.gz \
    "https://github.com/pyscf/pyscf-build-deps/blob/master/pyscf-2.14.1-deps.tar.gz?raw=true"
tar xzf deps.tar.gz
rm -f deps.tar.gz
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF -DXCFUN_MAX_ORDER=4 ..
make -j4
cd ..
rm -Rf build
cd ../..
