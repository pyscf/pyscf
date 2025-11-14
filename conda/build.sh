# You can run build.sh with bash -x -e. The -x makes it echo each command that is run
# and the -e makes it exit whenever a command in the script returns nonzero exit status.
set -x -e

# # Skip buliding libxc, libxcfun to speed up compiling
# export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF"
# wget http://www.sunqm.net/pyscf/files/bin/pyscf-2.0a-deps-openblas.tar.gz
# tar -C pyscf/lib -xzf pyscf-2.0a-deps-openblas.tar.gz
# # In this pre-build tarball libcint was linked against openblas. It should use
# # conda blas library in conda pkg. Remove libcint and compile it freshly
# find pyscf/lib/deps -name "*cint*" -exec rm {} \+
# rm pyscf-2.0-depsa-openblas.tar.gz

# C extensions must be installed with sequential BLAS library
# https://pyscf.org/install.html#using-optimized-blas
export CMAKE_CONFIGURE_ARGS="-DWITH_F12=OFF -DENABLE_SMD=ON -DBLA_VENDOR=Intel10_64lp_seq"

# env PYTHON not defined in certain conda-build version
# $PYTHON -m pip install . -vv
MAKEFLAGS="-j4" pip install -v --prefix=$PREFIX .
