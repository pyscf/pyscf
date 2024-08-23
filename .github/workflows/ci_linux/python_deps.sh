#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy!=1.5" h5py pytest pytest-cov pytest-timer
pip install pyberny
pip install --no-deps pyscf-dispersion

version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')
if [ $version != '3.12' ]; then
    pip install geometric
    pip install spglib
fi

#cppe
if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
    pip install cppe
fi
