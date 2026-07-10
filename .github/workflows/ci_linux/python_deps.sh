#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy!=1.5" h5py pytest pytest-cov pytest-timer
pip install git+https://github.com/jhrmnn/pyberny.git@36a4be9
pip install --no-deps pyscf-dispersion==1.5.0
pip install geometric

version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')

if [ $version == '3.12' ]; then
    pip install spglib
    pip install pytblis
    pip install git+https://github.com/sunqm/zquatev
fi

#cppe
#if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
#    pip install cppe
#fi
