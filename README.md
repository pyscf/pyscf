pyscf
=====

Python module for quantum chemistry

version 0.10

2015-2-4

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


Adding new features
-------------------
For developrs who has interests to add new features in this program,
there are few rules to follow

* New features first being placed in pyscf/future.
* Code at least should work under python-2.7, gcc-4.8.
* Not enforced, it's preferred
  - Compatibile with Python 2.6, 2.7, 3.2, 3.3, 3.4;
  - Following C89 standard for C code;
  - Using ctypes to bridge C/python functions, (to keep minimal dependence on third-party tools)
  - Avoid using other program language, to keep package light-weight
* Loose-coupling principle
  - Reinventing-wheel is encouraged if it reduces the coupling to the rest of the package.


Documentation
-------------

There is an online documentation  http://sunqm.net/pyscf.  And you can
also download the PDF version from  http://sunqm.net/pyscf/PySCF-0.10.pdf


Known problems
--------------

* Error message "Library not loaded: libcint.2.5.1.dylib" On OS X
  libcint.dylib is installed in  pyscf/lib/deps/lib  by default.  Add
  "/path/to/pyscf/lib/deps/lib"  to  `DYLD_LIBRARY_PATH`

* On debian-6, the system default BLAS library (libf77blas.so.3gf) might
  have bug in dsyrk function.  It occasionally results in NaN in mcscf
  solver.  To fix this, change to other BLAS vendors e.g. to MKL

        BLA_VENDOR=Intel10_64lp_seq cmake ..

* tests fail

  mcscf/test/test_addons.py    test_spin_square


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

