PySCF
=====

Python module for quantum chemistry

2016-03-08

* [Release 1.1 alpha-2](../../releases/latest)
* [Changelog](../master/CHANGELOG)
* [Documentation](http://www.pyscf.org) ([PDF](http://www.sunqm.net/pyscf/files/pdf/PySCF-1.1.pdf))
* [Installation](#installation)
* [Features](../master/FEATURES)


Installation
------------

* Prerequisites
    - Cmake 2.8 or higher
    - Python 2.6, 2.7, 3.2, 3.3, 3.4
    - Numpy 1.8.0 or higher
    - Scipy 0.10 or higher (0.12.0 or higher for python 3.3, 3.4)
    - h5py 1.3.0 or higher (requires HDF5 1.8.4 or higher)

* Compile core module

        cd lib
        mkdir build; cd build
        cmake ..
        make

* To make python be able to find pyscf, edit environment variable
  `PYTHONPATH`, e.g.  if pyscf is installed in /opt/pyscf

        export PYTHONPATH=/opt:$PYTHONPATH

* Use Intel MKL as BLAS library.  cmake with options
  `-DBLA_VENDOR=Intel10_64lp_seq`

        cmake -DBLA_VENDOR=Intel10_64lp_seq ..

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

* Using FCIQMC as the FCI solver for CASSCF.
      NECI (https://github.com/ghb24/NECI_STABLE)
  After installing the NECI, create a file future/fciqmc/settings.py
  to store the path where the NECI was installed.

* Using optimized integral library on X86 platform.  Qcint
  (https://github.com/sunqm/qcint.git) is a branch of libcint library.
  It is heavily optimized against X86_64 platforms.  To replace the
  default libcint library with qcint library, edit the URL of the
  integral library in lib/CMakeLists.txt file

        ExternalProject_Add(libcint
          GIT_REPOSITORY https://github.com/sunqm/qcint.git
          ...



Known problems
--------------

* Error message "Library not loaded: libcint.2.7.dylib" On OS X.
  libcint.dylib is installed in  pyscf/lib/deps/lib  by default.  Add
  "/path/to/pyscf/lib/deps/lib"  to  `DYLD_LIBRARY_PATH`

* Fails at runtime with error message
```
  OSError: ... mkl/lib/intel64/libmkl_avx.so: undefined symbol: ownLastTriangle_64fc
```

  This problem relates to MKL v11.1 on intel64 architecture.  Currently,
  there is no solution for the combination of Python + MKL 11.1 + AVX.
  You need either change to other MKL version (10.*, 11.0, 11.2) or
  disable mkl_avx:

        BLA_VENDOR=Intel10_64lp_seq cmake .. -DDISABLE_AVX=1

* tests fail

        mcscf/test/test_bz_df.py     test_mc2step_9o8e
        mcscf/test/test_addons.py    test_ucasscf_spin_square
        cc/test/test_h2o.py          test_h2o_without_scf


* Program exits with
```
AttributeError: ..../libri.so: undefined symbol: RInr_fill2c2e_sph
```

  It is caused by old version of libcint.  Remove the directory
  "pyscf/lib/deps" and rebuild pyscf to fix this problem.


* h5py installation.
  If you got problems to install the latest h5py package,  you can try
  the old releases:
  https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/
  https://github.com/h5py/h5py/archive/2.2.1.tar.gz



Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>

