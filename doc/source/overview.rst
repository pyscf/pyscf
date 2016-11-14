An overview of PySCF
********************

PySCF is an ab initio computational chemistry program implemented in
Python program language.  The package aims to provide a simple,
light-weight and efficient platform for electronic structure theory
developing and simulation.   The package provides a wide range of
functions to support the electronic structure mean-field and
post-mean-field calculations of finite size systems and extended systems
with periodic boundary condition.  Users can run simulations with input
script as the regular quantum chemistry package offers, or combine
primitive functions for new features, or even modify source code to
rapidly achieve certain requirements as they want.  Although most
functions are written in Python, the computation critical modules are
intensively optimized in C.  The package works as efficient as other
C/Fortran based quantum chemistry program.


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
