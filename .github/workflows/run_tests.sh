#!/usr/bin/env bash
export OMP_NUM_THREADS=1 
export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

echo 'pbc_tools_pbc_fft_engine = "NUMPY"' > .pyscf_conf.py
echo "dftd3_DFTD3PATH = './pyscf/lib/deps/lib'" >> .pyscf_conf.py
echo "scf_hf_SCF_mute_chkfile = True" >> .pyscf_conf.py

pytest pyscf/ -s -c setup.cfg \
    --cov-report xml --cov-report term --cov-config .coveragerc --cov pyscf
