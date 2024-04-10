#!/usr/bin/env bash
export OMP_NUM_THREADS=1 
export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

echo 'pbc_tools_pbc_fft_engine = "NUMPY"' > .pyscf_conf.py
echo "dftd3_DFTD3PATH = './pyscf/lib/deps/lib'" >> .pyscf_conf.py
echo "scf_hf_SCF_mute_chkfile = True" >> .pyscf_conf.py

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
# pytest-cov on Python 3.12 consumes huge memory
if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
  pytest pyscf/ -s -c pytest.ini \
    --cov-report xml --cov-report term --cov-config .coveragerc --cov pyscf
else
  pytest pyscf/ -s -c pytest.ini pyscf
fi
