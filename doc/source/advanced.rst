Miscellaneous
*************

Decoration pipe
===============

SCF
---
There are three decoration function for Hartree-Fock class
:func:`density_fit`, :func:`sfx2c`, :func:`newton` to apply density
fitting, scalar relativistic correction and second order SCF.
The different ordering of the three decoration operations have different
effects.  For example

.. literalinclude:: ../../examples/scf/23-decorate_scf.py

FCI
---
Direct FCI solver cannot guarantee the CI wave function to be the spin
eigenfunction.  Decoration function :func:`fci.addons.fix_spin_` can
fix this issue.

CASSCF
------
:func:`mcscf.density_fit`, and :func:`scf.sfx2c` can be used to decorate
CASSCF/CASCI class.  Like the ordering problem in SCF decoration
operation, the density fitting for CASSCF solver only affect the CASSCF
optimization procedure.  It does not change the 2e integrals for CASSCF
Hamiltonian.  For example

.. literalinclude:: ../../examples/mcscf/16-density_fitting.py


Customizing Hamiltonian
=======================

PySCF supports user-defined Hamiltonian for many modules.  To customize
Hamiltonian for Hartree-Fock, CASSCF, MP2, CCSD, etc, one need to replace the
methods :func:`get_hcore`, :func:`get_ovlp` and attribute :attr:`_eri` of SCF class for
new Hamiltonian.  E.g. the user-defined
Hamiltonian for Hartree-Fock

.. literalinclude:: ../../examples/scf/40-customizing_hamiltonian.py

and the user-defined Hamiltonian for CASSCF

.. literalinclude:: ../../examples/mcscf/40-customizing_hamiltonian.py

