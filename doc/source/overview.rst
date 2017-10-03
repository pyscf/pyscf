An overview of PySCF
********************

Python-based simulations of chemistry framework (PYSCF) is a general-purpose
electronic structure platform designed from the ground up to emphasize code
simplicity, so as to facilitate new method development and enable flexible
computational workflows. The package provides a wide range of tools to support
simulations of finite-size systems, extended systems with periodic boundary
conditions, low-dimensional periodic systems, and custom Hamiltonians, using
mean-field and post-mean-field methods with standard Gaussian basis functions.
To ensure ease of extensibility, PYSCF uses the Python language to implement
almost all of its features, while computationally critical paths are
implemented with heavily optimized C routines. Using this combined Python/C
implementation, the package is as efficient as the best existing C or Fortran-
based quantum chemistry programs.



Features
--------

.. include:: features.txt

* Interface to integral package `Libcint <https://github.com/sunqm/libcint>`_

* Interface to DMRG `CheMPS2 <https://github.com/SebWouters/CheMPS2>`_

* Interface to DMRG `Block <https://github.com/sanshar/Block>`_

* Interface to FCIQMC `NECI <https://github.com/ghb24/NECI_STABLE>`_

* Interface to XC functional library `XCFun <https://github.com/dftlibs/xcfun>`_

* Interface to XC functional library `Libxc <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc>`_


.. automodule:: pyscf


.. How to cite
.. -----------
.. Bibtex entry::
.. 
..   @Misc{PYSCF,
..     Title                    = {Python module for quantum chemistry program},
..     Author                   = {Qiming Sun},
..     HowPublished             = {\url{https://github.com/sunqm/pyscf.git}},
..     Year                     = {2014}
..   }
