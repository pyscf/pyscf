<div align="left">
  <img src="https://github.com/pyscf/pyscf-doc/blob/master/logo/pyscf-logo.png" height="80px"/>
</div>

Python-based Simulations of Chemistry Framework
===============================================
[![Build Status](https://travis-ci.org/pyscf/pyscf.svg?branch=master)](https://travis-ci.org/pyscf/pyscf)
[![codecov](https://codecov.io/gh/pyscf/pyscf/branch/master/graph/badge.svg)](https://codecov.io/gh/pyscf/pyscf)

2020-8-2

* [Stable release 1.7.4](https://github.com/pyscf/pyscf/releases/tag/v1.7.4)
* [Previous release 1.6.6](https://github.com/pyscf/pyscf/releases/tag/v1.6.6)
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
    - Scipy 0.10 or higher (0.12.0 or higher for python 3.4 - 3.8)
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


* Using DMRG as the FCI solver for CASSCF.  There are two DMRG solver
  interfaces available in pyscf.
      Block (https://sanshar.github.io/Block)
      CheMPS2 (https://github.com/SebWouters/CheMPS2)
  After installing the DMRG solver, create a file dmrgscf/settings.py
  to store the path where the DMRG solver was installed.

* Using FCIQMC as the FCI solver for CASSCF.
      NECI (https://github.com/ghb24/NECI_STABLE)
  After installing the NECI, create a file future/fciqmc/settings.py
  to store the path where the NECI was installed.

* Using optimized integral library on X86 platform.  [Qcint](https://github.com/sunqm/qcint.git)
  is a branch of libcint library.
  It is heavily optimized against X86_64 platforms.  To replace the
  default libcint library with qcint library, edit the URL of the
  integral library in lib/CMakeLists.txt file

        ExternalProject_Add(libcint
          GIT_REPOSITORY https://github.com/sunqm/qcint.git
          ...


Tutorials
---------
* A user-guide written in Ipython notebook can be found in https://github.com/nmardirossian/PySCF_Tutorial.
  This repository documents the basic structure of PySCF input script and the
  use of regular methods which were routinely executed in most quantum chemistry
  packages.  It also provides an implementation to drive PySCF program in a
  simple manner.
* Developer's tutorial can be found in the online documentation
  http://sunqm.github.io/pyscf/tutorial.html#tutorial and the repository above
  https://github.com/nmardirossian/PySCF_Tutorial/blob/master/dev_guide.ipynb


Citing PySCF
------------

The following paper should be cited in publications utilizing the PySCF program package:

PySCF: the Python-based Simulations of Chemistry Framework,
Q. Sun, T. C. Berkelbach, N. S. Blunt, G. H. Booth, S. Guo, Z. Li, J. Liu,
J. McClain, E. R. Sayfutyarova, S. Sharma, S. Wouters, G. K.-L. Chan (2018),
PySCF: the Python‐based simulations of chemistry framework.
WIREs Comput. Mol. Sci., 8: e1340. doi:[10.1002/wcms.1340](https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340)


Bug reports and feature requests
--------------------------------
Please submit tickets on the [issues](https://github.com/pyscf/pyscf/issues) page

