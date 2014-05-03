pyscf
=====

Python module for quantum chemistry SCF

version 0.1
2014-05-03


Installation
------------

* Prerequisites
    - Cmake version 2.8 or higher
    - Python version 2.6 or higher
    - Numpy version 1.6 or higher
    - Cython version 0.20 or higher (optional)
    - Libcint version 2.0 or higher
        https://github.com/sunqm/libcint
    - Libxc version 1.2.0 or higher (optional)
        http://www.tddft.org/programs/octopus/wiki/index.php/Libxc

* Compile core module

    cd lib
    mkdir build; cd build
    cmake ..
    make

* To make python be able to find pyscf, edit environment variable
  PYTHONPATH, e.g.  pyscf is installed in /opt/pyscf

    echo 'export PYTHONPATH=/opt/pyscf:$PYTHONPATH' >> ~/.bashrc

* Use Intel MKL as BLAS library.  cmake with options -DBLA_VENDOR=Intel10_64lp

    BLA_VENDOR=Intel10_64lp cmake ..

* If cmake complains "Could NOT find libcint" or "Could NOT find libxc",
  set CMAKE_PREFIX_PATH to the directories where libcint and libxc are
  installed.  E.g. if libcint is installed in /opt/libcint, libxc is
  installed in /opt/libxc,

    CMAKE_PREFIX_PATH=/opt/libcint:/opt/libxc:$CMAKE_PREFIX_PATH cmake ..


Bug report
----------
Qiming Sun <osirpt.sun@gmail.com>

