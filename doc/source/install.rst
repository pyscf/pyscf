.. _installing:

Installation
************

You may already have `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_
and `h5py <http://www.h5py.org/>`_ installed.  If not, you can install
them from any Python package managers (`Pypi <https://pypi.python.org/>`_,
`conda <http://conda.pydata.org/>`_).  Here we recommend to use the
integrated science platform `Anaconda <https://www.continuum.io/downloads#linux>`_.
(with `conda-cmake <https://anaconda.org/anaconda/cmake>`_).

You can download the latest PySCF release version
`1.2 <https://github.com/sunqm/pyscf/releases/tag/v1.2>`_ or the
develment branch from github

  $ git clone https://github.com/sunqm/pyscf

Build the C extensions in :file:`pyscf/lib`::

  $ cd pyscf/lib
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

It will automatically download the analytical GTO integral library
`libcint <https://github.com/sunqm/libcint.git>`_ and the DFT exchange
correlation functional library `libxc <http://www.tddft.org/programs/Libxc>`_
and `xcfun <https://github.com/dftlibs/xcfun.git>`_.  Finally, to make Python
be able to find pyscf package, add the **parent directory** of pyscf to
:code:`PYTHONPATH`, e.g. assuming pyscf is put in ``/home/abc``::

  export PYTHONPATH=/home/abc:$PYTHONPATH

To ensure the installation is successed, start a Python shell, and type::

  >>> import pyscf

If you got errors like::

  ImportError: No module named pyscf

It's very possible that you put ``/home/abc/pyscf`` in the :code:`PYTHONPATH`.
You need to remove the ``/pyscf`` in that string and try import
``pyscf`` in the python shell again.

For Mac OsX user, you may get an import error if your OsX version is
10.11 or later::

    OSError: dlopen(xxx/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.2.8.dylib
    Referenced from: xxx/pyscf/lib/libcgto.dylib
    Reason: unsafe use of relative rpath libcint.2.8.dylib in xxx/pyscf/lib/libao2mo.dylib with restricted binary

This is caused by the RPATH 
It can be fixed by running the script ``pyscf/lib/_runme_to_fix_dylib_osx10.11.sh`` in ``pyscf/lib``::
 
    cd pyscf/lib
    sh _runme_to_fix_dylib_osx10.11.sh


.. note::

  RPATH has been built in the dynamic library.  This may cause library loading
  error on some systems.  You can run ``pyscf/lib/_runme_to_remove_rpath.sh`` to
  remove the rpath code from the library head.  Another workaround is to set
  ``-DCMAKE_SKIP_RPATH=1`` and ``-DCMAKE_MACOSX_RPATH=0`` in cmake command line.
  When the RPATH was removed, you need to add ``pyscf/lib`` and
  ``pyscf/lib/deps/lib`` in ``LD_LIBRARY_PATH``.


Installation without network
============================

If you get problem to download the external libraries on your computer, you can
manually build the libraries, as shown in the following instructions.  First,
you need to install libcint, libxc or xcfun libraries.
`libcint cint3 branch <https://github.com/sunqm/libcint/tree/cint3>`_
and `xcfun stable-1.x branch <https://github.com/dftlibs/xcfun/tree/stable-1.x`_
are required by PySCF.  They can be downloaded from github::

    $ git clone https://github.com/sunqm/libcint.git
    $ cd libcint
    $ git checkout origin/cint3
    $ cd .. && tar czf libcint.tar.gz libcint

    $ git clone https://github.com/sunqm/xcfun.git
    $ cd xcfun
    $ git checkout origin/stable-1.x
    $ cd .. && tar czf xcfun.tar.gz xcfun

libxc-2.2.* can be found in http://octopus-code.org/wiki/Main_Page .
Assuming ``/opt`` is the place where these libraries will be installed, these
packages should be compiled with the flags::

    $ tar xvzf libcint.tar.gz
    $ cd libcint
    $ mkdir build && cd build
    $ cmake -DWITH_F12=1 -DWITH_RANGE_COULOMB=1 -DWITH_COULOMB_ERF=1 \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt -DCMAKE_INSTALL_LIBDIR:PATH=lib ..
    $ make && make install

    $ tar xvzf libxc-2.2.2.tar.gz
    $ cd libxc-2.2.2
    $ mkdir build && cd build
    $ ../configure --prefix=/opt --libdir=/opt/lib --enable-shared --disable-fortran LIBS=-lm
    $ make && make install

    $ tar xvzf xcfun.tar.gz
    $ cd xcfun
    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1 -DXC_MAX_ORDER=3 -DXCFUN_ENABLE_TESTS=0 \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt -DCMAKE_INSTALL_LIBDIR:PATH=lib ..
    $ make && make install

Next compile PySCF::

    $ cd pyscf/pyscf/lib
    $ mkdir build && cd build
    $ cmake -DBUILD_LIBCINT=0 -DBUILD_LIBXC=0 -DBUILD_XCFUN=0 -DCMAKE_INSTALL_PREFIX:PATH=/opt ..
    $ make

Finally update the ``PYTHONPATH`` environment for Python interpreter.


.. _installing_blas:

Using optimized BLAS
====================

The default installation does not need to provide external linear
algebra libraries.  It's possible that the setup script only find and
link to the slow BLAS/LAPACK libraries.  You can install the package
with other BLAS venders instead of the default one to improve the
performance,  eg MKL (it can provide 10 times speedup in many modules)::

  $ cd pyscf/lib/build
  $ cmake -DBLA_VENDOR=Intel10_64lp_seq ..
  $ make

If you are using Anaconda as your Python-side platform, you can link PySCF
to the MKL library coming with Anaconda package::

  $ export MKLROOT=/path/to/anaconda2
  $ export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
  $ cd pyscf/lib/build
  $ cmake -DBLA_VENDOR=Intel10_64lp_seq ..
  $ make

You can link to other BLAS libraries by setting ``BLA_VENDOR``, eg
``BLA_VENDOR=ATLAS``, ``BLA_VENDOR=IBMESSL``.  Please refer to `cmake mannual
<http://www.cmake.org/cmake/help/v3.0/module/FindBLAS.html>`_ for more details
of the use of ``FindBLAS`` macro.

If the cmake ``BLA_VENDOR`` cannot find the right BLAS library as you expected,
you can assign the libraries to the variable ``BLAS_LIBRARIES`` in
:file:`lib/CMakeLists.txt`::

  set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
  set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
  set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
  set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")


.. _installing_qcint:

Using optimized integral library
================================

The default integral library used by PySCF is
libcint (https://github.com/sunqm/libcint).  To ensure the
compatibility on various high performance computer systems, PySCF does
not use the fast integral library by default.  For X86-64 platforms,
libcint library has an efficient implementation Qcint
https://github.com/sunqm/qcint.git
which is heavily optimized against SSE3 instructions.
To replace the default libcint library with qcint library, edit the URL
of the integral library in lib/CMakeLists.txt file::

  ExternalProject_Add(libcint
     GIT_REPOSITORY
     https://github.com/sunqm/qcint.git
     ...


Offline installation
====================

Compiling PySCF will automatically download and compile
`libcint <https://github.com/sunqm/libcint.git>`_,
`libxc <http://www.tddft.org/programs/Libxc>`_
and `xcfun <https://github.com/dftlibs/xcfun.git>`_.   If the
compilation breaks due to the failure of download or compilation of
these packages, you can manually download and install them then install
PySCF offline.  ``pyscf/lib/deps`` is the directory where PySCF places
the external libraries.  PySCF will bypass the compilation of the
external libraries if they were existed in that directory.  In the PySCF
offline compilcation mode, you need install these external libraries to
this directory.  Followings are the relevant compiling flags for these
libraries.

Libcint::

    cd /path/to/libcint
    mkdir build
    cd build
    cmake -DWITH_RANGE_COULOMB=1 -DCMAKE_INSTALL_PREFIX:PATH=/path/to/pyscf/lib/deps -DCMAKE_INSTALL_LIBDIR:PATH=lib ..
    make && make install

XcFun::

    cd /path/to/xcfun
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1 -DXC_MAX_ORDER=3 -DXCFUN_ENABLE_TESTS=0 -DCMAKE_INSTALL_PREFIX:PATH=/path/to/pyscf/lib/deps -DCMAKE_INSTALL_LIBDIR:PATH=lib ..
    make && make install

LibXC::

    cd /path/to/libxc
    mkdir build
    cd build
    ../configure --prefix=/path/to/pyscf/lib/deps --libdir=/path/to/pyscf/lib/deps/lib --enable-shared --disable-fortran LIBS=-lm
    make && make install


.. _installing_plugin:

Plugins
=======

DMRG solver
-----------
Density matrix renormalization group (DMRG) implementations Block
(http://chemists.princeton.edu/chan/software/block-code-for-dmrg) and
CheMPS2 (http://sebwouters.github.io/CheMPS2/index.html)
are efficient DMRG solvers for ab initio quantum chemistry problem.
`Installing Block <http://sanshar.github.io/Block/build.html>`_ requires
C++11 compiler.  If C++11 is not supported by your compiler, you can
register and download the precompiled Block binary from
http://chemists.princeton.edu/chan/software/block-code-for-dmrg.
Before using the Block or CheMPS2, you need create a config file
future/dmrgscf/settings.py  (as shown by settings.py.example) to store
the path where the DMRG solver was installed.

FCIQMC
------
NECI (https://github.com/ghb24/NECI_STABLE) is FCIQMC code developed by
George Booth and Ali Alavi.  PySCF has an interface to call FCIQMC
solver NECI.  To use NECI, you need create a config file
future/fciqmc/settings.py to store the path where NECI was installed.

Libxc
-----
By default, building PySCF will automatically download and install
`Libxc 2.2.2 <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`_.
:mod:`pyscf.dft.libxc` module provided a general interface to access Libxc functionals.

Xcfun
-----
By default, building PySCF will automatically download and install
latest xcfun code from https://github.com/dftlibs/xcfun.
:mod:`pyscf.dft.xcfun` module provided a general interface to access Libxc
functionals.

XianCI
------
XianCI is a spin-adapted MRCI program.  "Bingbing Suo" <bsuo@nwu.edu.cn>
is the main developer of XianCI program.

