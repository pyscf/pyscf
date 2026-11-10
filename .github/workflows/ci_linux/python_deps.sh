#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy!=1.5" h5py pytest pytest-cov pytest-xdist pytest-timeout
pip install pyberny
pip install --no-deps pyscf-dispersion==1.7.0
pip install geometric
pip install trexio

version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')

if [ $version == '3.12' ]; then
    pip install pytblis
fi

# Include symmetry and quaternion eigensolver support in the nightly suite,
# independently of the Python version selected by that workflow.
if [ "$version" = '3.12' ] || [ "${INSTALL_NIGHTLY_DEPS:-false}" = 'true' ]; then
    pip install spglib
    pip install git+https://github.com/sunqm/zquatev
    pip install basis-set-exchange
fi

#cppe
#if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
#    pip install cppe
#fi
