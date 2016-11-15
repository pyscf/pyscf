.. _installing:

Installation
************

You may already have `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_
and `h5py <http://www.h5py.org/>`_ installed.  If not, you can use
``python-pypi`` to install numpy, scipy and h5py::

  $ pip install --target=/path/to/python/libs numpy
  $ pip install --target=/path/to/python/libs scipy
  $ pip install --target=/path/to/python/libs h5py

or install the integrated science platform `anaconda <https://www.continuum.io/downloads#linux>`_.
You can download the latest release version
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

It can be fixed by running the script ``pyscf/lib/_runme_to_fix_dylib_osx10.11.sh`` in ``pyscf/lib``::
 
    cd pyscf/lib
    sh _runme_to_fix_dylib_osx10.11.sh


.. _installing_blas:

Using optimized BLAS
====================

The Linear algebra libraries have significant affects on the performance
of the PySCF package.  The default installation does not require to
provide external linear algebra libraries.  It's possible that the setup
script only find and link to the slow BLAS/LAPACK libraries.  You can
install the package with other BLAS venders instead of the default one
to improve the performance,  eg MKL (it can provide 10 times speedup in
many modules)::

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

