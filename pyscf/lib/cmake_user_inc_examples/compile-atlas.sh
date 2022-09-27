#!/bin/bash

# this file must be copied in pyscf/lib
# Python path binary if necessary (Anaconda, intelPython)
export PATH=/PATH_TO_ANACONDA/anaconda3/bin:$PATH

ml purge
ml load module to load if necessary
# For Atlas (12/2017)
# SIESTA/4.1-b3-gimkl-2017b CMake/3.9.4-gimkl-2017b

export CC=gcc
export FC=gfortran
export CXX=g++

export MKLROOT=/PATH_TO_MKL_DIR
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

cd build
rm -r *
cmake -DBLA_VENDOR=Intel10_64lp_seq ..
make VERBOSE=1
