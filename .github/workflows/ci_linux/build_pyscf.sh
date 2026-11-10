#!/usr/bin/env bash

set -e -o pipefail

cd ./pyscf/lib
if [ "${PYSCF_CI_PROFILE:-}" = "manylinux-aarch64" ]; then
    curl --fail --location -o deps.tar.gz "https://media.githubusercontent.com/media/pyscf/pyscf-build-deps/master/pyscf-2.2-aarch64-deps.tar.gz"
    # Keep XCFun, but build libcint and LibXC at the versions in CMakeLists.txt.
    tar xzf deps.tar.gz deps/include/XCFun deps/lib64/libxcfun.so deps/lib64/libxcfun.so.2
    cmake_args=(-DBUILD_LIBXC=ON -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=ON)
    build_jobs=2
else
    curl --fail --location "https://media.githubusercontent.com/media/pyscf/pyscf-build-deps/master/pyscf-2.14.1-deps.tar.gz" | tar xzf -
    cmake_args=(-DBUILD_LIBXC=OFF -DBUILD_XCFUN=ON -DBUILD_LIBCINT=OFF -DXCFUN_MAX_ORDER=4)
    build_jobs=4
fi
mkdir build; cd build
cmake "${cmake_args[@]}" ..
make -j"$build_jobs"
cd ..
rm -Rf build
cd ../..
