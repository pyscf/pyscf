#!/usr/bin/env bash
export OMP_NUM_THREADS=1 
export PYTHONPATH=$(pwd):$PYTHONPATH 

echo $PYTHONPATH
lldb ./pyscf/lib/libxc_itrf.dylib
ls ./pyscf/lib/deps/lib
lldb ./pyscf/lib/deps/lib/libxc.5.dylib
python -c "from pyscf import lib; lib.load_library('libxc_itrf')"
python -c "from pyscf import dft; print(dft.libxc)"
