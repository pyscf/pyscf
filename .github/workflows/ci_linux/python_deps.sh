#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy!=1.5" h5py pytest pytest-cov pytest-timer pytest-xdist pytest-timeout
pip install pyberny
# pyscf-dispersion 1.6.0 has not been released on PyPI yet.
pip install --no-deps git+https://github.com/pyscf/dispersion.git@78ae53d4c40a632aa4fb86be5315db1c8eb01e0e
pip install geometric
pip install trexio

version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')

if [ $version == '3.12' ]; then
    pip install spglib
    pip install pytblis
    pip install git+https://github.com/sunqm/zquatev
    pip install basis-set-exchange
fi

#cppe
#if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
#    pip install cppe
#fi
