.. _mix_mol:

Mixing PBC and molecular modules
********************************
 
The post-HF methods, as a standalone numerical solver, do not require the
knowledge of the boundary condition.  The calculations of finite-size systems
and extend systems are distinguished by the boundary condition of integrals (and
basis).  The same post-HF solver can be used for both the finite-size problem
and the periodic boundary problem if they have the similar Hamiltonian
structure.

In PySCF, many molecular post-HF solvers has two implementations: incore and
outcore versions.  They are differed by the treatments on the 2-electron
integrals.  The incore solver takes the :attr:`_eri` (or :attr:`with_df`, see
:ref:`mol_df`) from the underlying mean-field object as the two-electron
interaction part of the Hamiltonian while the outcore solver generates the
2-electron integrals (with free boundary condition) on the fly.
To use the molecular post-HF solvers in PBC code, we need ensure the incore
version solver being called.

Generating :attr:`_eri` in mean-filed object is the straightforward way to
trigger the incore post-HF solver.  If the allowed memory is big enough to
hold the entire 2-electron integral array, the gamma point HF solver always
generates and holds this array.  A second choice is to set :attr:`incore_anyway`
in ``cell`` which forces the program generating and holding :attr:`_eri` in
mean-field object.

.. note::

  If the problem is big, :attr:`incore_anyway` may overflow the available
  physical memory.

Holding the full integral array :attr:`_eri` in memory limits the problem size
one can treat.  Using the density fitting object :attr:`with_df` to hold the
integrals can overcome this problem.  This architecture has been bound to PBC
and molecular mean-field modules.  But the relevant post-HF density fitting
solvers are still in development thus this feature is not available in PySCF 1.2
or older.

Aside from the 2-electron integrals, there are some attributes and methods
required by the post-HF solver.  They are :meth:`get_hcore`, and
:meth:`get_ovlp` for 1-electron integrals, :attr:`_numint`, :attr:`grids` for
the numerical integration of DFT exchange-correlation functionals.  They are all
overloaded in PBC mean-field object to produce the PBC integrals. 


Examples
--------

.. literalinclude:: ../../../examples/pbc/12-gamma_point_post_hf.py
