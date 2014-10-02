pyscf
=====

Python module for quantum chemistry

version 0.5

2014-10-01

Pyscf is an open-source suite of quantum chemistry program.  The program
aims to provide a simple, light-weight and efficient platform for quantum
chemistry code developing and calculation.  The program is developed in
the principle of
* Easy to install, to use, to extend and to be embedded;
* Minimal requirements on libraries (No Boost, MPI) and computing
  resources (perhaps losing efficiency to reduce I/O);
* 90/10 Python/C, only computational hot spots were written in C;
* 90/10 functional/OOP, unless performance critical, functions are pure.


Installation
------------

* Prerequisites
    - Cmake 2.8 or higher
    - Python 2.6 or higher
    - Numpy 1.6 or higher
    - Scipy 0.11 or higher
    - HDF5 1.8.4 or higher
    - h5py 1.3.0 or higher
    - Libcint 2.0.4 or higher
        https://github.com/sunqm/libcint
    - Libxc 2.0.0 or higher (optional for DFT)
        http://www.tddft.org/programs/octopus/wiki/index.php/Libxc

* Compile core module

    cd lib
    mkdir build; cd build
    cmake ..
    make

* To make python be able to find pyscf, edit environment variable
  PYTHONPATH, e.g.  if pyscf is installed in /opt/pyscf

    export PYTHONPATH=/opt:$PYTHONPATH

* Use Intel MKL as BLAS library.  cmake with options -DBLA_VENDOR=Intel10_64lp

    BLA_VENDOR=Intel10_64lp cmake ..

  If cmake still cannot find MKL, just define BLAS_LIBRARIES in CMakeLists.txt

    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")

* If cmake complains "Could NOT find libcint" or "Could NOT find libxc",
  set CMAKE_PREFIX_PATH to the directories where libcint and libxc are
  installed.  E.g. if libcint is installed in /opt/libcint, libxc is
  installed in /opt/libxc,

    CMAKE_PREFIX_PATH=/opt/libcint:/opt/libxc:$CMAKE_PREFIX_PATH cmake ..


Known problems
--------------


Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>


Changes
-------
Version 0.1 (2014-05-03):
  * Setup pyscf

Version 0.2 (2014-05-08):
  * Integral transformation

Version 0.3 (2014-07-03):
  * Change import layout

Version 0.4 (2014-08-17):
  * Module "future" for upcoming functions
  * One-line command to run QC calculation with pyscf
  * Fix bug of AO to MO transformation in OpenMP environment

Version 0.5 (2014-10-01):
  * Change basis format
  * Remove Cython dependence
  * Upgrade dft to use libxc-2.0.0
  * Add DFT, FCI, CASSCF, HF-gradients (NR and R), HF-NMR (NR and R)
