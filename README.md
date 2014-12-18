pyscf
=====

Python module for quantum chemistry

version 0.7

2014-11-12

Pyscf is a python module for quantum chemistry program.  The module
aims to provide a simple, light-weight and efficient platform for
quantum chemistry code developing and calculation.  The program is
developed in the principle of
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
    - Numpy 1.6.2 or higher (1.6.1 has bug in einsum)
    - Scipy 0.10 or higher
    - HDF5 1.8.4 or higher
    - h5py 1.3.0 or higher

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

  If cmake is still not able to find MKL, just define BLAS_LIBRARIES in CMakeLists.txt

    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")

* Using DMRG as the FCI solver for CASSCF.  There are two DMRG solver
  interfaces avaialbe in pyscf.
      Block (https://github.com/sanshar/Block)
      CheMPS2 (https://github.com/SebWouters/CheMPS2)
  After installing the DMRG solver, create a file future/dmrgscf/settings.py
  to store the path where the DMRG solver was installed.


Adding new features
-------------------
For developrs who has interests to add new features in this program,
there are few rules to follow

* New features first being placed in pyscf/future.
* Code at least should work under python-2.7, gcc-4.8.
* Not enforced, it's preferred
  - Compatibile with 2.5 - 3.3 for Python code;
  - Following C89 standard for C code;
  - Using ctypes to bridge C/python functions, (to keep minimal dependence on third-party tools)
  - Avoid using other program language, to keep package light-weight
* Loose-coupling principle
  - Reinventing-wheel is encouraged if it reduces the coupling to the rest of the package.


Known problems
--------------
* Error message "Library not loaded: libcint.2.3.0.dylib" On OS X
  libcint.dylib is installed in  pyscf/lib/deps/lib  by default.  Add
  "/path/to/pyscf/lib/deps/lib"  to  DYLD_LIBRARY_PATH


Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>


Changes log
-----------
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

Version 0.6 (2014-10-17):
  * Fix bug in dhf
  * add future/lo for localized orbital

Version 0.7 (2014-11-12):
  * Fix memory leaks
  * Runtime keywords checking
  * Add MP2 density matrix
  * Add FCI based on uhf integrals
  * Add CCSD

Version 0.8 (2014-12-?):
  * Support OS X
  * MCSCF for triplet
  * Add symmetry support for MCSCF
  * Add 2-step DMRGSCF, using Block and ChemPS2 as FCI solver
  * Add ROHF
