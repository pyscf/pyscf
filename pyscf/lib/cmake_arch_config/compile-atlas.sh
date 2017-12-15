#!/bin/bash

export PATH=/scratch/mbarbry/anaconda3/bin:$PATH

ml purge
ml load SIESTA/4.1-b3-gimkl-2017b CMake/3.9.4-gimkl-2017b

export CC=gcc
export FC=gfortran
export CXX=g++
#export MKLROOT=/scratch/mbarbry/anaconda3
export MKLROOT=/sNow/easybuild/SLES11SP3/haswell/software/imkl/2017.3.196-iimpi-2017b/mkl
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

cd build
rm -r *
cmake -DBLA_VENDOR=Intel10_64lp_seq ..
make VERBOSE=1
