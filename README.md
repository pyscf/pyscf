pyscf
=====

Python module for quantum chemistry

Version 1.0 beta

2015-8-2

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
    - Python 2.6, 2.7, 3.2, 3.3, 3.4
    - Numpy 1.6.2 or higher (1.6.1 has bug in einsum)
    - Scipy 0.10 or higher
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

        BLA_VENDOR=Intel10_64lp_seq cmake ..

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


Adding new features
-------------------
For developrs who has interests to add new features in this program,
there are few rules to follow

* Code at least should work under python-2.7, gcc-4.8.
* Not enforced
  - Compatibile with Python 2.6, 2.7, 3.2, 3.3, 3.4;
  - Following C89 standard for C code;
  - Using ctypes to bridge C/python functions, (to keep minimal dependence on third-party tools)
  - Avoid using other program language, to keep package light-weight


Documentation
-------------

There is an online documentation  http://www.pyscf.org.  And you can
also download the PDF version from  http://www.pyscf.org/PySCF-1.0.pdf


Known problems
--------------

* Error message "Library not loaded: libcint.2.5.1.dylib" On OS X
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
  mcscf/test/test_addons.py    test_spin_square
  cc/test/test_h2o.py          test_h2o_without_scf


* Program exits with
```
AttributeError: ..../libri.so: undefined symbol: RInr_fill2c2e_sph
```

  It is caused by old version of libcint.  Remove the directory
  "pyscf/lib/deps" and rebuild pyscf to fix this problem.


```
Exception AttributeError: "'NoneType' object has no attribute 'byref'" in
<bound method VHFOpt.__del__ of <pyscf.scf._vhf.VHFOpt object at 0x2b52390>> ignored
```
  It was observed when pyscf is used with inspectors like profiler, pdb
  etc.




Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>

