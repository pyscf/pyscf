#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install numpy "scipy==1.1" h5py nose nose-exclude nose-timer nose-cov codecov
pip install pyberny geometric

#cppe
version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')
if [ $version != '2.7' ] && [ $version != '3.5' ]; then
    pip install cppe
fi
