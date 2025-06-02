#!/usr/bin/env bash
export OMP_NUM_THREADS=4
export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

mkdir -p pyscftmpdir
echo 'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"' > .pyscf_conf.py
echo "dftd3_DFTD3PATH = './pyscf/lib/deps/lib'" >> .pyscf_conf.py
echo "scf_hf_SCF_mute_chkfile = True" >> .pyscf_conf.py
echo 'TMPDIR = "./pyscftmpdir"' >> .pyscf_conf.py

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
# pytest-cov on Python 3.12 consumes huge memory
if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
  pytest pyscf/ -s -c pytest.ini \
    --cov-report xml --cov-report term --cov-config .coveragerc --cov pyscf
else
  pytest pyscf/ -s -c pytest.ini pyscf
fi

pytest_status=$?

num_tmpfiles="$(ls -1 pyscftmpdir | wc -l)"
echo "There are "$num_tmpfiles" leftover temporary files"
rm -rf pyscftmpdir

# Test fails if pytest failed or if temporary files were left over.
if test "$num_tmpfiles" -gt 0 || test "$pytest_status" -ne 0; then
  exit 1
fi
