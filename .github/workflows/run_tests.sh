#!/usr/bin/env bash

# By default the tests run in parallel processes (pytest-xdist), one BLAS/OpenMP
# thread each. Most tests are too small to keep 4 BLAS threads busy, so spreading
# the work over processes uses the runner's cores far better than threading it.
#
# The three knobs below let a workflow job reshape that. The multi-threaded job
# raises OMP_NUM_THREADS so the OpenMP kernels are actually exercised with more
# than one thread, and narrows TEST_PATHS to keep the runtime down.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

# Number of pytest worker processes. "auto" is one per core, "0" disables xdist
# entirely. Dial back if the workers start competing for memory.
PYTEST_JOBS=${PYTEST_JOBS:-auto}
# Which tests to run.
TEST_PATHS=${TEST_PATHS:-pyscf}
# Additional command-line arguments override pytest.ini selection or defaults,
# e.g. -k "not _skip" includes high-cost tests in the nightly workflow.

if [ "$PYTEST_JOBS" = "0" ]; then
  PARALLEL=""
else
  # loadfile keeps every test in a file on the same worker. Several tests write
  # chkfiles named relative to the working directory, and a few rely on module
  # level setUpModule fixtures that are expensive to rebuild per worker.
  PARALLEL="-n $PYTEST_JOBS --dist loadfile"
fi

# try to reduce oversubscription caused by OPENBLAS_THREAD_SERVER
export OPENBLAS_THREAD_TIMEOUT=4
export OMP_WAIT_POLICY=passive

mkdir -p pyscftmpdir
echo 'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"' > .pyscf_conf.py
echo "dftd3_DFTD3PATH = './pyscf/lib/deps/lib'" >> .pyscf_conf.py
echo "scf_hf_SCF_mute_chkfile = True" >> .pyscf_conf.py
echo 'TMPDIR = "./pyscftmpdir"' >> .pyscf_conf.py

# pytest-cov (and the codecov upload of its report) runs only in the nightly job
# that sets RUN_COVERAGE. Other jobs run without coverage instrumentation to
# reduce runtime and memory under xdist.
if [ "$RUN_COVERAGE" = "true" ]; then
  python -u .github/workflows/ci_linux/pytest_live.py -s -c pytest.ini --timeout=900 $PARALLEL \
    --durations=20 \
    --cov-report xml --cov-report term --cov-config .coveragerc --cov pyscf \
    $TEST_PATHS "$@"
else
  python -u .github/workflows/ci_linux/pytest_live.py -s -c pytest.ini --timeout=900 $PARALLEL --durations=20 $TEST_PATHS "$@"
fi

pytest_status=$?

num_tmpfiles="$(ls -1 pyscftmpdir | wc -l)"
echo "There are "$num_tmpfiles" leftover temporary files"
rm -rf pyscftmpdir

# Preserve pytest's status after cleanup, including collection and usage errors.
if test "$pytest_status" -ne 0; then
  exit "$pytest_status"
fi
if test "$num_tmpfiles" -gt 0; then
  exit 1
fi
