#!/usr/bin/env bash
cd ./pyscf/lib
wget https://github.com/fishjojo/pyscf-deps/raw/master/pyscf-1.7.5-deps-macos-10.14.tar.gz
tar xzf pyscf-1.7.5-deps-macos-10.14.tar.gz
mkdir build; cd build
cmake -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF ..
make -j4
cd ../../..
