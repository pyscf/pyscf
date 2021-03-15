#!/usr/bin/env bash
cd ./pyscf/lib
curl http://www.sunqm.net/pyscf/files/bin/pyscf-2.0-deps.tar.gz | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF ..
make -j4
cd ..
rm -Rf build
cd ../..
