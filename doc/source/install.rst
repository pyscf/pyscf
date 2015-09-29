.. _installing:

Installation
************

You may already have `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_
and `h5py <http://www.h5py.org/>`_ installed.  If not, you can use pip
to install numpy, scipy and h5py::

  $ pip install --target=/path/to/python/libs numpy
  $ pip install --target=/path/to/python/libs scipy
  $ pip install --target=/path/to/python/libs h5py

Download the latest version of `pyscf <https://github.com/sunqm/pyscf.git/>`_::

  $ git clone https://github.com/sunqm/pyscf

Now you need build the C extensions in :file:`pyscf/lib`::

  $ cd pyscf/lib
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

It will automatically download the analytical GTO integral library
``libcint`` https://github.com/sunqm/libcint.git and the DFT
exchange correlation functional library ``libxc``
http://www.tddft.org/programs/Libxc.  Finally, to make Python find pyscf
package, add the **parent directory** of pyscf to :code:`PYTHONPATH`,
e.g. assuming pyscf is put in ``/home/abc``::

  export PYTHONPATH=/home/abc:$PYTHONPATH

To ensure the installation is successed, start a Python shell, and type::

  >>> import pyscf

If you got errors like::

  ImportError: No module named pyscf

It's very possible that you put ``/home/abc/pyscf`` in the :code:`PYTHONPATH`.
You need to remove the ``/pyscf`` in that string and try import
``pyscf`` in the python shell again.


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
  $ BLA_VENDOR=Intel10_64lp_seq cmake ..
  $ make

You can link to other BLAS libraries by setting ``BLA_VENDOR``, eg
``BLA_VENDOR=ATLAS``, ``BLA_VENDOR=IBMESSL``.  Please refer to `cmake
mannual <http://www.cmake.org/cmake/help/v3.0/module/FindBLAS.html>`_
for more details of the use of ``FindBLAS`` macro.

If the cmake ``BLA_VENDOR`` cannot detect the right BLAS library as you
expected, you can simply assign the libraries to the variable
``BLAS_LIBRARIES`` in :file:`lib/CMakeLists.txt`::

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


.. _installing_plugin:

Plugins
=======

DMRG solver
-----------

There are two DMRG solver interfaces avaialbe in PySCF:
Block (http://chemists.princeton.edu/chan/software/block-code-for-dmrg)
and CheMPS2 (http://sebwouters.github.io/CheMPS2/index.html).
Before using the DMRG, you need create a config file
future/dmrgscf/settings.py  (as shown by settings.py.example) to store
the path where the DMRG solver was installed.

FCIQMC
------
PySCF has an interface to call FCIQMC solver NECI
(https://github.com/ghb24/NECI_STABLE).  To use NECI, you need
create a config file future/fciqmc/settings.py to store the path where
NECI was installed.

