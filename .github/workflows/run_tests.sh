#!/usr/bin/env bash
# The tests run in parallel processes (pytest-xdist), one BLAS/OpenMP thread
# each. Most tests are too small to keep 4 BLAS threads busy, so spreading the
# work over processes uses the runner's cores far better than threading it.
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

# Number of pytest worker processes. "auto" is one per core; override to dial
# back if the workers start competing for memory.
PYTEST_JOBS=${PYTEST_JOBS:-auto}
# loadfile keeps every test in a file on the same worker. Several tests write
# chkfiles named relative to the working directory, and a few rely on module
# level setUpModule fixtures that are expensive to rebuild per worker.
PARALLEL="-n $PYTEST_JOBS --dist loadfile"

mkdir -p pyscftmpdir
echo 'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"' > .pyscf_conf.py
echo "dftd3_DFTD3PATH = './pyscf/lib/deps/lib'" >> .pyscf_conf.py
echo "scf_hf_SCF_mute_chkfile = True" >> .pyscf_conf.py
echo 'TMPDIR = "./pyscftmpdir"' >> .pyscf_conf.py

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
# pytest-cov on Python 3.12 consumes huge memory
if [ "$RUNNER_OS" == "Linux" ] && [ $version != "3.12" ]; then
  pytest -s -c pytest.ini $PARALLEL \
    --durations=20 \
    --cov-report xml --cov-report term --cov-config .coveragerc --cov pyscf
else
  pytest -s -c pytest.ini $PARALLEL --durations=20 pyscf
fi

pytest_status=$?

num_tmpfiles="$(ls -1 pyscftmpdir | wc -l)"
echo "There are "$num_tmpfiles" leftover temporary files"
rm -rf pyscftmpdir

# Test fails if pytest failed or if temporary files were left over.
if test "$num_tmpfiles" -gt 0 || test "$pytest_status" -ne 0; then
  exit 1
fi
