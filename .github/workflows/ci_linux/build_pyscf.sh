#!/usr/bin/env bash
cd ./pyscf/lib
curl http://www.sunqm.net/pyscf/files/bin/pyscf-1.7.5-deps.tar.gz | tar xzf -
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF ..
make -j4
cd ..
rm -Rf build
rm -f pyscf-1.7.5-deps.tar.gz
cd ../..
