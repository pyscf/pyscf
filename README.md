pyscf
=====

Python module for quantum chemistry SCF

version 0.3

2014-07-03


Installation
------------

* Prerequisites
    - Cmake 2.8 or higher
    - Python 2.6 or higher
    - Numpy 1.6 or higher
    - HDF5 1.8.4 or higher
    - h5py 1.3.0 or higher
    - Cython 0.20 or higher (optional)
    - Libcint 2.0 or higher
        https://github.com/sunqm/libcint
    - Libxc 1.2.0 or higher (optional)
        http://www.tddft.org/programs/octopus/wiki/index.php/Libxc

* Compile core module

    cd lib
    mkdir build; cd build
    cmake ..
    make

* To make python be able to find pyscf, edit environment variable
  PYTHONPATH, e.g.  pyscf is installed in /opt/pyscf

    export PYTHONPATH=/opt:$PYTHONPATH

* Use Intel MKL as BLAS library.  cmake with options -DBLA_VENDOR=Intel10_64lp

    BLA_VENDOR=Intel10_64lp cmake ..

  If cmake still cannot find MKL, just define BLAS_LIBRARIES CMakeLists.txt

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
* Runtime Warning

    .../scf/hf.py:26: RuntimeWarning: compiletime version 2.6 of module
    '_vhf' does not match runtime version xxx

  The program is not mattered for Python 2.7.  To get rid of this
  warning message, you need install Cython and recompile

    lib/vhf/_vhf.pyx
    lib/ao2mo/_ao2mo.pyx.


Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>


Changes
-------
Version 0.1 (2014-05-03):
  * Setup pyscf

Version 0.2 (2014-05-08):
  * AO to MO transformation

Version 0.3 (2014-07-03):
  * Change import layout

