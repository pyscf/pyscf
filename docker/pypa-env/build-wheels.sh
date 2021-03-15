#!/usr/bin/env bash

set -e -x

src=${GITHUB_WORKSPACE:-/src/pyscf}
dst=${GITHUB_WORKSPACE:-/src/pyscf}/linux-wheels
mkdir -p /root/wheelhouse $src/linux-wheels

if [ "$#" -gt 0 ]; then
  export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF"
  curl $1 | tar -C $src/pyscf/lib -xzf -
else
  export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF"
fi

# Compile wheels
for PYVERSION in cp36-cp36m cp37-cp37m cp38-cp38 cp39-cp39; do
    PYBIN=/opt/python/$PYVERSION/bin
    "${PYBIN}/pip" wheel -v --no-deps --no-clean -w /root/wheelhouse $src

    # Bundle external shared libraries into the wheels
    whl=`ls wheelhouse/pyscf-*-$PYVERSION-linux_x86_64.whl`
    auditwheel -v repair "$whl" --lib-sdir /lib -w $dst
done
