<div align="left">
  <img src="https://github.com/sunqm/pyscf/blob/master/doc/logo/pyscf-logo.png" height="80px"/>
</div>

Python-based Simulations of Chemistry Framework
===============================================

2018-03-23

* [Stable release 1.4.5](https://github.com/sunqm/pyscf/releases/tag/v1.4.5)
* [1.5 alpha](https://github.com/sunqm/pyscf/releases/tag/v1.5-alpha)
* [Changelog](../master/CHANGELOG)
* [Documentation](http://www.pyscf.org)
* [Installation](#installation)
* [Features](../master/FEATURES)


Installation
------------

* Prerequisites
    - Cmake 2.8 or higher
    - Python 2.6, 2.7, 3.4 or higher
    - Numpy 1.8.0 or higher
    - Scipy 0.10 or higher (0.12.0 or higher for python 3.4 - 3.6)
    - h5py 2.3.0 or higher (requires HDF5 1.8.4 or higher)

* Compile core module

        cd pyscf/lib
        mkdir build; cd build
        cmake ..
        make

  Note during the compilation, external libraries (libcint, libxc, xcfun) will
  be downloaded and installed.  If you want to disable the automatic
  downloading, this [document](http://sunqm.github.io/pyscf/install.html#installation-without-network)
  shows how to manually build these packages and PySCF C libraries.

* To export PySCF to Python, you need to set environment variable `PYTHONPATH`.
  E.g.  if PySCF is installed in /opt, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/pyscf:$PYTHONPATH

* Using Intel MKL as BLAS library.  Enabling the cmake options
  `-DBLA_VENDOR=Intel10_64lp_seq` when executing cmake

        cmake -DBLA_VENDOR=Intel10_64lp_seq ..

  If cmake does not find MKL, you can define BLAS_LIBRARIES in CMakeLists.txt

        set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
        set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
        set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
        set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")

* Using DMRG as the FCI solver for CASSCF.  There are two DMRG solver
  interfaces avaialbe in pyscf.
      Block (http://chemists.princeton.edu/chan/software/block-code-for-dmrg)
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

* Using pyberny (https://github.com/azag0/pyberny) as geometry optimizer.
  After downloading pyberny

      git clone https://github.com/azag0/pyberny /path/to/pyberny

  edit the environment variable to make pyscf find pyberny

      export PYTHONPATH=/path/to/pyberny:$PYTHONPATH


Known problems
--------------

* Error message "Library not loaded: libcint.3.0.dylib" On OS X.

  libcint.dylib is installed in  pyscf/lib/deps/lib  by default.  Add
  "/path/to/pyscf/lib/deps/lib"  to  `DYLD_LIBRARY_PATH`

* On Mac OSX, error message of "import pyscf"
```
  OSError: dlopen(xxx/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.3.0.dylib
  Referenced from: xxx/pyscf/lib/libcgto.dylib
  Reason: unsafe use of relative rpath libcint.3.0.dylib in xxx/pyscf/lib/libao2mo.dylib with restricted binary
```

  It is only observed on OSX 10.11.  One solution is to manually modify the runtime path

        $ install_name_tool -change libcint.3.0.dylib xxx/pyscf/lib/deps/lib/libcint.3.0.dylib xxx/pyscf/lib/libcgto.dylib
        $ install_name_tool -change libcint.3.0.dylib xxx/pyscf/lib/deps/lib/libcint.3.0.dylib xxx/pyscf/lib/libcvhf.dylib
        $ install_name_tool -change libcint.3.0.dylib xxx/pyscf/lib/deps/lib/libcint.3.0.dylib xxx/pyscf/lib/libao2mo.dylib
        ...

  Running script pyscf/lib/_runme_to_fix_dylib_osx10.11.sh  can patch
  all required dylib files.


* runtime error message
```
  OSError: ... mkl/lib/intel64/libmkl_avx.so: undefined symbol: ownLastTriangle_64fc
```
  or
```
  MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.
```

  This is a MKL 11.* bug when MKL is used with "dlopen" function.
  Preloading MKL libraries can solve this problem on most systems:

```
  export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_sequential.so:$MKLROOT/lib/intel64/libmkl_core.so
```

  or 

```
  export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx.so:$MKLROOT/lib/intel64/libmkl_core.so
```


* h5py installation.

  If you got problems to install the latest h5py package,  you can try
  the old releases:
  https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/
  https://github.com/h5py/h5py/archive/2.3.1.tar.gz


* If you are using Intel compiler (version 16, 17), compilation may be stuck at
```
[ 95%] Building C object CMakeFiles/cint.dir/src/stg_roots.c.o
```

  This code is used by F12 integrals only.  If you do not need F12 methods,
  the relevant compilation can be disabled, by searching `DWITH_F12` in file
  lib/CMakeLists.txt  and setting it to `-DWITH_F12=0`.



Citing PySCF
------------

The following paper should be cited in publications utilizing the PySCF program package:

* PySCF: the Python-based Simulations of Chemistry Framework,
  Q. Sun, T. C. Berkelbach, N. S. Blunt, G. H. Booth, S.  Guo, Z. Li, J. Liu,
  J. McClain, E. R. Sayfutyarova, S. Sharma, S. Wouters, G. K.-L. Chan,
  WIREs Comput Mol Sci 2017, DOI: 10.1002/wcms.1340


Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>

