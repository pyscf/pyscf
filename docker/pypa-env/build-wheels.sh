#!/usr/bin/env bash

set -e -x

src=${GITHUB_WORKSPACE:-/src/pyscf}
dst=${GITHUB_WORKSPACE:-/src/pyscf}/linux-wheels
mkdir -p /root/wheelhouse $src/linux-wheels

if [ "$#" -gt 0 ]; then
  export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF -DBUILD_LIBCINT=OFF -DENABLE_SMD=ON"
  curl -L $1 | tar -C $src/pyscf/lib -xzf -
else
  export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF -DENABLE_SMD=ON"
fi

# In certain versions of auditwheel, some .so files was excluded.
sed -i '/            if basename(fn) not in needed_libs:/s/basename.*libs/1/' /opt/_internal/pipx/venvs/auditwheel/lib/python*/site-packages/auditwheel/wheel_abi.py

# Compile wheels
PYVERSION=cp39-cp39
PYBIN=/opt/python/$PYVERSION/bin
"${PYBIN}/pip" wheel -v --no-deps --no-clean -w /root/wheelhouse $src

# Bundle external shared libraries into the wheels
whl=`ls /root/wheelhouse/pyscf-*_x86_64.whl`
auditwheel -v repair "$whl" --lib-sdir /lib -w $dst
