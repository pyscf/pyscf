Library of tools for Managing the numerical atomic orbitals (NAO)
===============================================

2017-08-07

Installation
------------

* Compile core module with NAO support

        cd pyscf/lib
        cp nao/cmakelists_examples/CMakeLists.txt.gnu CMakeLists.txt
        mkdir build
        cd build
        cmake ..
        make

  Note during the compilation, external libraries (libcint, libxc, xcfun) will
  be downloaded and installed.  If you want to disable the automatic
  downloading, this [document](http://sunqm.github.io/pyscf/install.html#installation-without-network)
  is an instruction for manually building these packages.

* Testing of NAO support 

  After a succesful compilation, you may want to reproduce the test results. 
  A batch testing is possible by commanding

        cd pyscf
        python -m unittest discover test

  Tests are also executable on one-by-one basis.
  For example:
        python test/test_0001_system_vars.py

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

