Library of tools for Managing the numerical atomic orbitals (NAO)
===============================================

2019-12-08

Installation
------------

* Dependence

    * Fortran, C, Python
    * CMake
    * BLAS, Lapack, FFTW
    * numpy, scipy 

* Default compilation of pySCF's low-level libraries with NAO support is possible with custom
  architecture file.
  
        cd pyscf/lib
        cp cmake_user_inc_examples/cmake.user.inc-nao-gnu cmake.user.inc
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


Once installed, you will need to export the following PATH to your bash prompt

    export PYTHONPATH=~/pyscf:${PYTHONPATH}
    export LD_LIBRARY_PATH=~/pyscf/pyscf/lib/deps/lib:${LD_LIBRARY_PATH}
  

Testing of NAO support 
----------------------

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
  
  Anaconda's scipy/numpy installations use it's own MKL library (by Intel,
  provides high-speed subroutines for BLAS/LAPACK/FFTW). The (fortran) library
  libnao also profits from these low-level libraries. Moreover, we found it
  mandatory to link against the same MKL library if OpenMP parallelization is
  to be used. In order to link against Anaconda's MKL we have to modify CMakeLists.txt
  file entering the path for BLAS and LAPACK libraries. One can spare finding
  FFTW library. An example of such CMakeLists.txt file can be found at
  nao/cmakelists_examples/ directory.

        apt-get update
        apt-get upgrade
        apt-get -y install git wget curl gcc gfortran build-essential liblapack-dev libfftw3-dev make cmake zlib1g-dev

        # Download anaconda if not already installed (optional)
        wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
        bash Anaconda3-2019.10-Linux-x86_64.sh
        bash

        # Anaconda comes with MKL. We are then gonna to use the MKL present in
        # Anaconda to install PySCF-NAO
        # To do so, we create a mk/lib directory in anaconda directory to separate it
        # from the other libraries
        mkdir -p anaconda3/lib/mkl/lib
        ln -s ~/anaconda3/lib/libmkl_* ~/anaconda3/lib/mkl/lib/

        # Install Pyscf-NAO
        export CC=gcc && export FC=gfortran && export CXX=g++
        git clone https://github.com/cfm-mpc/pyscf.git
        cd pyscf
        git fetch
        git checkout nao2
        cd pyscf/lib
        cp cmake_user_inc_examples/cmake.user.inc-singularity.anaconda.gnu.mkl cmake.arch.inc

        # Edit the cmake.arch.inc file and spcify the proper path to the
        # MKLROOT variable

        export LD_LIBRARY_PATH=~/anaconda3/lib/mkl/lib:{LD_LIBRARY_PATH}
        mkdir build && cd build
        cmake .. && make
 
* IntelPython

  The statements above related to Anaconda's usage apply to a similar extend also in the case of IntelPython.
  In case of IntelPython, the MKL subdirectory contains also a file for GNU treating libmkl_intel_thread.so, but 
  in fact we failed to run the libraries compiled/linked with gfortran and called from intelpython.
  Be sure also to "activate" the IntelPython before starting the interpreter. Activation is done by "sourcing"
  the script activate in the IntelPython's bin subdirectory. There is an example of CMakeLists.txt file. 

        cd pyscf/lib
        cp cmake_user_inc_examples/cmake.user.inc-intelpython-ifort cmake.user.inc
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

* Numba:
  Some Numba version are known to give problems.
  * 0.41: gives a segfault in `ls_contributing_numba`


Citing PySCF/NAO
------------

When the iterative TDDFT is used, please cite

* PySCF-NAO: An efficient and flexible implementation of linear response time-dependent density functional theory with numerical atomic orbitals, P. Koval, M. Barbry and D. Sanchez-Portal, Computer Physics Communications, 2019, 10.1016/j.cpc.2018.08.004
* A Parallel Iterative Method for Computing Molecular Absorption Spectra, P. Koval, D. Foerster, and O. Coulaud, J. Chem. Theo. Comput. 2010, 10.1021/ct100280x

See also citing instructions in the main README.

Bug report
----------
Koval Peter <koval.peter@gmail.com>

