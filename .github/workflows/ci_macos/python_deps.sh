#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy<=1.1" h5py pytest pytest-cov pytest-timer
pip install pyberny geometric

version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')

#scipy
if [ $version == '3.6' ]; then
    pip install 'scipy<1.2'
elif [ $version == '3.7' ]; then
    pip install 'scipy<1.2'
else
    pip install scipy
fi

#cppe
if [ $version != '2.7' ] && [ $version != '3.5' ]; then
    pip install cppe
fi
