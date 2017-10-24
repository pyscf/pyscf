Library of tools for Managing the numerical atomic orbitals (NAO)
===============================================

2017-08-27

Installation
------------

* Default compilation of pySCF's low-level libraries with NAO support is possible with custom
  architecture file.
  
        cd pyscf/lib
        cp cmake_arch_config/cmake.arch.inc-gnu cmake.arch.inc
        mkdir build
        cd build
        export FC=gfortran   # (just to be sure)
        cmake ..
        make

  Note during the compilation, external libraries (libcint, libxc, xcfun) will
  be downloaded and installed.  If you want to disable the automatic
  downloading, this [document](http://sunqm.github.io/pyscf/install.html#installation-without-network)
  is an instruction for manually building these packages.
  The shared object files are put in the directory   pyscf/lib.

* Using the same build directory for repetitive compilation
  
  Usually there is no problem to recompile the libraries after an update,
  however, if one intents to use different compilers/interpreters/linked libraries
  the file CMakeCache.txt must be deleted and cmake started again.
        
        cd build
        rm CMakeCache.txt
        cmake ..
        make 

  If some linkage problem with LibXC or xcfun libraries arises, then maybe the version of these libraries is now changed to a higher one. Delete the directory pyscf/lib/deps to start downloading/compilation from scratch.
  
* Testing of NAO support 

  After a succesful compilation, you may want to reproduce the test results. 
  A batch testing is possible by commanding

        cd pyscf/nao                                # (directory with python code for NAO)
        python -m unittest discover test            # (pyscf/nao/test contains examples)

  Tests are also executable on one-by-one basis.
  For example:
        python test/test_0001_system_vars.py

Peak performance builds/runs
------------

* Anaconda
  
  Anaconda's scipy/numpy installations use it's own MKL library (by Intel, provides high-speed subroutines for BLAS/LAPACK/FFTW)
  The (fortran) library libnao also profits from these low-level libraries. Moreover, we found it mandatory to link against 
  the same MKL library if OpenMP parallelization is to be used. In order to link against Anaconda's MKL we have to modify
  CMakeLists.txt file entering the path for BLAS and LAPACK libraries. One can spare finding FFTW library. An example of such
  CMakeLists.txt file can be found at nao/cmakelists_examples/ directory. Moreover, we have found that only Intel compiler 
  could be used to link against OpenMP-enabled compilations. In the Anaconda's MKL subdirectory (anaconda2/pkgs/mkl-2017.0.3-0/lib/)
  there are only libmkl_intel_thread.so and libmkl_sequential.so files, i.e. a library supporting GNU threads is not provided.
  Therefore, unfortunately, the top-speed runs can be only achieved with Intel Fortran compiler installed.
  In order to use Intel Fortran compiler (ifort) during compilation of libnao, please "export" the shell variable FC before 
  doing first cmake in the build directory. With this in mind, and provided that the Intel Fortran compiler is working and 
  findable as "ifort", the sequence of commands to build the library would be as following:
  
        cd pyscf/lib
        cp cmake_arch_config/cmake.arch.inc-anaconda-gnu cmake.arch.inc
        rm *.so                # (just to be sure)
        mkdir anaconda_build
        cd anaconda_build
        export FC=ifort
        cmake ..
        make
  
  The shared object files are put in the directory   pyscf/lib.      
  Before starting the interpreter, you might want to activate it
        
        source /path/to/anaconda2/bin/activate

* IntelPython

  The statements above related to Anaconda's usage apply to a similar extend also in the case of IntelPython.
  In case of IntelPython, the MKL subdirectory contains also a file for GNU treating libmkl_intel_thread.so, but 
  in fact we failed to run the libraries compiled/linked with gfortran and called from intelpython.
  Be sure also to "activate" the IntelPython before starting the interpreter. Activation is done by "sourcing"
  the script activate in the IntelPython's bin subdirectory. There is an example of CMakeLists.txt file. 

        cd pyscf/lib
        cp cmake_arch_config/cmake.arch.inc-intelpython-ifort cmake.arch.inc
        rm *.so                              # (just to be sure)
        mkdir intelpython_ifort
        cd mkl_intelpython_ifort
        export FC=ifort
        cmake ..
        make
  
  The shared object files are put in the directory   pyscf/lib.
  Before starting the interpreter, you might want to activate it
        
        source /path/to/intelpython2/bin/activate

Known problems
--------------

* Some tests are not verified with small differences when using Anaconda/IntelPython while everything passes when using a system's python installation:

  Probable different BLAS/LAPACK libraries are used by libnao and numpy. See installation instructions and Peak performance builds/runs.
  
* Most of the tests are failing. 

  Probable BLAS/LAPACK/FFTW calls could not be resolved from libnao. See installation instructions and Peak performance builds/runs.

* Some linkage problem with libxc or xcfun arises

  Maybe the optimal version of these libraries is now changed to a higher one. Delete the directory pyscf/lib/deps to start downloading/compilation from scratch.


Citing PySCF/NAO
------------

When the iterative TDDFT is used, please cite

@article{iter_method,
author = {Koval, Peter and Foerster, Dietrich and Coulaud, Olivier},
title = {A Parallel Iterative Method for Computing Molecular Absorption Spectra},
journal = {J. Chem. Theo. Comput.},
volume = {6},
number = {9},
pages = {2654--2668},
year = {2010},
doi = {10.1021/ct100280x},
URL = {http://pubs.acs.org/doi/abs/10.1021/ct100280x},
abstract = { We describe a fast parallel iterative method for computing molecular 
absorption spectra within TDDFT linear response and using the LCAO method. We use a 
local basis of “dominant products” to parametrize the space of orbital products that 
occur in the LCAO approach. In this basis, the dynamic polarizability is computed 
iteratively within an appropriate Krylov subspace. The iterative procedure uses a 
matrix-free GMRES method to determine the (interacting) density response. The 
resulting code is about 1 order of magnitude faster than our previous full-matrix 
method. This acceleration makes the speed of our TDDFT code comparable with codes 
based on Casida’s equation. The implementation of our method uses hybrid MPI and 
OpenMP parallelization in which load balancing and memory access are optimized. 
To validate our approach and to establish benchmarks, we compute spectra of 
large molecules on various types of parallel machines. The methods developed 
here are fairly general, and we believe they will find useful applications in 
molecular physics/chemistry, even for problems that are beyond TDDFT, such as 
organic semiconductors, particularly in photovoltaics. }
}

See also citing instructions in the main README.

Bug report
----------
Koval Peter <koval.peter@gmail.com>

