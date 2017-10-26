.. _installing:

Installation
************

We provide three ways to install PySCF package.

Installation with conda
=======================

If you have `Conda <https://conda.io/docs/>`_ 
(or `Anaconda <https://www.continuum.io/downloads#linux>`_)
environment, PySCF package can be installed with the command as bellow::

  $ conda install -c pyscf pyscf

Installation with pip
=====================

You have to first install the dependent libraries (due to the missing of
build-time dependency in pip `PEP 518 <https://www.python.org/dev/peps/pep-0518/>`_)::
 
  $ pip install numpy scipy h5py

Then install PySCF::

  $ pip install pyscf

.. note::
  BLAS library is required to install PySCF library.  In some systems, the
  installation can automatically detect the installed BLAS libraries in the
  system and choose one for the program.  If BLAS library is existed
  in the system but the install script couldn't find it, you can specify the
  BLAS library either through the environment ``LDFLAGS``, eg
  ``LDFLAGS="-L/path/to/blas -lblas" pip install pyscf`` or the environment
  variable ``PYSCF_INC_DIR``, eg
  ``PYSCF_INC_DIR=/path/to/blas:/path/to/other/lib pip install``.

.. note::
  libxc library is not available in the PyPI repository.  pyscf.dft module is
  not working unless the libxc library was installed in the system.  You can
  download libxc library from http://octopus-code.org/wiki/Libxc:download
  (note you need to add --enable-shared when compiling the libxc library).
  Before calling pip, the path where the libxc library is installed needs to be
  added to the environment variable ``PYSCF_INC_DIR``, eg
  ``export PYSCF_INC_DIR=/path/to/libxc``.

.. note::
  Depending on the operator systems, you may fail to ``pip install h5py`` due to
  the missing of Python header files and HDF5 libraries.  For Linux OS, you can
  get Python header files by installing ``apt-get install python-dev``
  (``yum install python-devel`` for redhat) and HDF5 libraries
  ``apt-get install libhdf5-dev`` (``yum install hdf5-devel`` for redhat).


Manual installation from github repo
====================================

You can manually install PySCF from the PySCF github repo.
Manual installation requires `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_
and `h5py <http://www.h5py.org/>`_ libraries.
You can download the latest PySCF version (or the development branch) from github::

  $ git clone https://github.com/sunqm/pyscf
  $ cd pyscf
  $ git checkout dev  # optional if you'd like to try out the development branch

Build the C extensions in :file:`pyscf/lib`::

  $ cd pyscf/lib
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

This will automatically download the analytical GTO integral library `libcint
<https://github.com/sunqm/libcint.git>`_ and the DFT exchange correlation
functional libraries `libxc <http://www.tddft.org/programs/Libxc>`_ and `xcfun
<https://github.com/dftlibs/xcfun.git>`_.  Finally, to make Python find
the :code:`pyscf` package, add the top-level :code:`pyscf` directory (not
the :code:`pyscf/pyscf` subdirectory) to :code:`PYTHONPATH`.  For example, if
:code:`pyscf` is installed in ``/opt``, :code:`PYTHONPATH` should be like::

  export PYTHONPATH=/opt/pyscf:$PYTHONPATH

To ensure the installation is successful, start a Python shell, and type::

  >>> import pyscf

For Mac OS X/macOS, you may get an import error if your OS X/macOS version is
10.11 or newer::

    OSError: dlopen(xxx/pyscf/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.3.0.dylib
    Referenced from: xxx/pyscf/pyscf/lib/libcgto.dylib
    Reason: unsafe use of relative rpath libcint.3.0.dylib in xxx/pyscf/pyscf/lib/libcgto.dylib with restricted binary

This is caused by the incorrect RPATH.  Script
``pyscf/lib/_runme_to_fix_dylib_osx10.11.sh`` in ``pyscf/lib`` directory can be
used to fix this problem::
 
    cd pyscf/lib
    sh _runme_to_fix_dylib_osx10.11.sh


.. note::

  RPATH has been built in the dynamic library.  This may cause library loading
  error on some systems.  You can run ``pyscf/lib/_runme_to_remove_rpath.sh`` to
  remove the rpath code from the library head.  Another workaround is to set
  ``-DCMAKE_SKIP_RPATH=1`` and ``-DCMAKE_MACOSX_RPATH=0`` in cmake command line.
  When the RPATH was removed, you need to add ``pyscf/lib`` and
  ``pyscf/lib/deps/lib`` in ``LD_LIBRARY_PATH``.

Last, it's recommended to set a scratch directory for PySCF.  The default scratch
directory is controlled by environment variable :code:`PYSCF_TMPDIR`.  If it's
not specified, the system temporary directory :code:`TMPDIR` will be used as the
scratch directory.


Installation without network
============================

If you have problems to download the external libraries on your computer, you can
manually build the libraries, as shown in the following instructions.  First,
you need to install libcint, libxc or xcfun libraries.
`libcint cint3 branch <https://github.com/sunqm/libcint/tree/cint3>`_
and `xcfun stable-1.x branch <https://github.com/dftlibs/xcfun/tree/stable-1.x>`_
are required by PySCF.  They can be downloaded from github::

    $ git clone https://github.com/sunqm/libcint.git
    $ cd libcint
    $ git checkout origin/cint3
    $ cd .. && tar czf libcint.tar.gz libcint

    $ git clone https://github.com/sunqm/xcfun.git
    $ cd xcfun
    $ git checkout origin/stable-1.x
    $ cd .. && tar czf xcfun.tar.gz xcfun

libxc-3.* can be found in http://octopus-code.org/wiki/Main_Page or
`here <http://sunqm.net/pyscf/files/src/libxc-3.0.0.tar.gz>`_.
Assuming ``/opt`` is the place where these libraries will be installed, these
packages should be compiled with the flags::

    $ tar xvzf libcint.tar.gz
    $ cd libcint
    $ mkdir build && cd build
    $ cmake -DWITH_F12=1 -DWITH_RANGE_COULOMB=1 -DWITH_COULOMB_ERF=1 \
        -DCMAKE_INSTALL_PREFIX:PATH=/opt -DCMAKE_INSTALL_LIBDIR:PATH=lib ..
    $ make && make install

    $ tar xvzf libxc-3.0.0.tar.gz
    $ cd libxc-0.0.0
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

The default installation tries to find BLAS libraries automatically. This
automated setup script may link the code to slow BLAS libraries.  You can
compile the package with other BLAS vendors to improve performance, for example
the Intel Math Kernel Library (MKL), which can provide 10x speedup in many
modules::

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
libcint (https://github.com/sunqm/libcint).  This integral library was
implemented in the model that ensures the compatibility on various high
performance computer systems.  For X86-64 platforms, libcint library has an
efficient counterpart Qcint (https://github.com/sunqm/qcint)
which is heavily optimized against X86 SIMD instructions (AVX-512/AVX2/AVX/SSE3).
To replace the default libcint library with qcint library, edit the URL
of the integral library in lib/CMakeLists.txt file::

  ExternalProject_Add(libcint
     GIT_REPOSITORY
     https://github.com/sunqm/qcint.git
     ...


.. _installing_plugin:

Plugins
=======

nao
---
:mod:`pyscf/nao` module includes the basic functions of numerical atomic orbitals
(NAO) and the (nao based) TDDFT methods.  This module was contributed by Marc
Barbry and Peter Koval.  You can enable this module with a cmake flag::

    $ cmake -DENABLE_NAO=1 ..

More information of the compilation can be found in :file:`pyscf/lib/nao/README.md`.

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

