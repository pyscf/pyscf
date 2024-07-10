.. _pbc_scf:

pbc.scf --- Mean-field with periodic boundary condition
*******************************************************
This module is an analogy to molecular :mod:`pyscf.scf` module to handle
mean-filed calculation with periodic boundary condition.

Gamma point and single k-point calculation
==========================================
The usage of gamma point Hartree-Fock program is very close to that of the
molecular program.  In the PBC gamma point calculation, one needs initialize
:class:`Cell` object and the corresponding :class:`pyscf.pbc.scf.hf.RHF` class::

    from pyscf.pbc import gto, scf
    cell = gto.M(
        atom = '''H     0.      0.      0.    
                  H     0.8917  0.8917  0.8917''',
        basis = 'sto3g',
        h = '''
        0       1.7834  1.7834
        1.7834  0       1.7834
        1.7834  1.7834  0     ''',
        gs = [10]*3,
        verbose = 4,
    )
    mf = scf.RHF(cell).run()

Comparing to the :class:`pyscf.scf.hf.RHF` object for molecular calculation,
the PBC-HF calculation with :class:`pyscf.pbc.scf.hf.RHF` or
:class:`pyscf.pbc.scf.uhf.UHF` has three differences

* :class:`psycf.pbc.scf.hf.RHF` is the single k-point PBC HF class.  By default,
  it creates the gamma point calculation.  You can change to other k-point by
  setting the :attr:`kpt` attribute::

    mf = scf.RHF(cell)
    mf.kpt = cell.get_abs_kpts([.25,.25,.25])  # convert from scaled kpts
    mf.kernel()

* The exchange integrals of the PBC Hartree-Fock method has slow convergence
  with respect to the number of k-points.  Proper treatments for the divergent
  part of exchange integrals can effectively improve the convergence.  Attribute
  :mod:`exxdive` is used to control the method to handle exchange divergent
  term.  The default ``exxdiv='ewald'`` is favored in most scenario.  However,
  if the molecular post-HF methods was mixed with the gamma point HF method (see
  :ref:`mix_to_mol`, you might need set ``exxdiv=None`` to get consistent total
  energy (see :ref:`exxdiv`).

* In the finite-size system, one can obtain right answer without considering the
  model to evaluate 2-electron integrals.  But the integral scheme might need to
  be updated in the PBC calculations.  The default integral scheme is accurate
  for pseudo-potential.  In the all-electron calculation, you may need change
  the :attr:`with_df` attribute to mixed density fitting (MDF) method for better
  accuracy (see :ref:`with_df`).  Here is an example to update :attr:`with_df`

.. literalinclude:: ../../../examples/pbc/11-gamma_point_all_electron_scf.py


.. _mix_to_mol:

Mixing with molecular program for post-HF methods
-------------------------------------------------
The gamma point HF code adopts the same code structure, the function and
method names and the arguments' convention as the molecule SCF code.
This desgin allows one mixing PBC HF object with the existed molecular post-HF
code for PBC electron correlation problems.  A typical molecular post-HF
calculation starts from the finite-size HF method with the :class:`Mole`
object::

    from pyscf import gto, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz')
    mf = scf.RHF(mol).run()

    from pyscf import cc
    cc.CCSD(mf).run()

The PBC gamma point post-HF calculation requires the :class:`Cell` object and
PBC HF object::

    from pyscf.pbc import gto, scf
    cell = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz',
                 h=numpy.eye(3)*2, gs=[10,10,10])
    mf = scf.RHF(cell).run()

    from pyscf import cc
    cc.CCSD(mf).run()

The differences are the the ``mol`` or ``cell`` object to create and the
``scf`` module to import.  With the system-specific mean-field object, one
can carray out various post-HF methods (MP2, Coupled cluster, CISD, TDHF,
TDDFT, ...) using the same code for finite-size and extended systems.
See :ref:`mix_mol` for more details of the interface between PBC and molecular
modules.


k-point sampling
================

Newton solver
-------------

Smearing
--------


.. _exxdiv:

Exchange divergence treatment
=============================
The PBC Hartree-Fock has slow convergence of exchange integral with respect to
the number of k-points.  In the single k-point calculation,
Generally, exxdiv leads to a shift in the total energy and the spectrum of
orbital energy.  It should not affect the following correlation energy in the
post-HF calculation.  In practice, when the gamma-point calculation is mixed
with molecular program eg the FCI solver, the exxdiv attribute may leads to
inconsistency in the total energy.


.. _with_df:

:attr:`with_df` for density fitting
===================================

Placing the :attr:`with_df` attribute in SCF object to get the compatibility to
molecule DF-SCF methods.


Stability analysis
==================


Program reference
=================

.. automodule:: pyscf.pbc.scf.hf
   :members:

.. automodule:: pyscf.pbc.scf.uhf
   :members:

.. automodule:: pyscf.pbc.scf.khf
   :members:

.. automodule:: pyscf.pbc.scf.kuhf
   :members:

