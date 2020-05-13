MPI wrapper from GPAW containing Blacs/Scalapack routines
=========================================================

2017-09-19

Original code source
--------------------

https://gitlab.com/gpaw/gpaw/tree/master/c

Installation
------------

* Compile core module with MPI support

        export CC=mpicompiler (mpicc for example)
        cd pyscf/lib
        mkdir build
        cd build
        cmake -DMPI_BUILD=ON ..
        make

  Note during the compilation, external libraries (libcint, libxc, xcfun) will
  be downloaded and installed.  If you want to disable the automatic
  downloading, this [document](http://sunqm.github.io/pyscf/install.html#installation-without-network)
  is an instruction for manually building these packages.

* Testing of MPI support 

  TODO
