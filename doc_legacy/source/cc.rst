.. _cc:

cc --- Coupled cluster
**********************

.. module:: cc
   :synopsis: Computing coupled cluster energies and properties
.. sectionauthor:: Qiming Sun <osirpt.sun@gmail.com>.

The :mod:`cc` module implements the coupled cluster (CC) model to compute
energies, analytical nuclear gradients, density matrices, excited states, and
relevant properties.

To compute the CC energy, one first needs to perform a mean-field calculation using
the mean-field module :mod:`scf`.  The mean-field object defines the Hamiltonian
and the problem size, which are used to initialize the CC object::

    from pyscf import gto, scf, cc
    mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    mycc.kernel()

Unrelaxed density matrices are evaluated in the MO basis::

    dm1 = mycc.make_rdm1()
    dm2 = mycc.make_rdm2()

The CCSD(T) energy can be obtained by::

    from pyscf.cc import ccsd_t
    print(ccsd_t.kernel(mycc, mycc.ao2mo())[0])

Gradients are available::

    from pyscf.cc import ccsd_grad
    from pyscf import grad
    grad_e = ccsd_grad.kernel(mycc)
    grad_n = grad.grad_nuc(mol)
    grad = grad_e + grad_nuc

Excited states can be calculated with ionization potential (IP), electron affinity (EA),
and electronic excitation (EE) equation-of-motion (EOM) CCSD::

    mycc = cc.RCCSD(mf)
    mycc.kernel()
    e_ip, c_ip = mycc.ipccsd(nroots=1)
    e_ea, c_ea = mycc.eaccsd(nroots=1)
    e_ee, c_ee = mycc.eeccsd(nroots=1)

    mycc = cc.UCCSD(mf)
    mycc.kernel()
    e_ip, c_ip = mycc.ipccsd(nroots=1)
    e_ea, c_ea = mycc.eaccsd(nroots=1)
    e_ee, c_ee = mycc.eeccsd(nroots=1)

All CC methods have two implementations.  One is simple and highly readable (suffixed
by ``_slow`` in the filename) and the other is extensively optimized for
computational efficiency.
All code in the ``_slow`` versions is structured as close as
possible to the formulas documented in the literature.  Pure Python/numpy
data structures and functions are used so that explicit memory management is avoided.
It is easy to make modifications or develop new methods based on the slow
implementations.

The computationally efficient (outcore) version is the default implementation
for the CC module.  In this implementation, the CPU usage, memory footprint,
memory efficiency, and IO overhead are carefully considered.  To keep a small
memory footprint, most integral tensors are stored on disk.  IO is one of the
main bottlenecks in this implementation.  Two techniques are used to reduce
the IO overhead.  One is the asynchronized IO to overlap the computation and
reading/writing of the 4-index tensors.  The other is AO-driven for the
contraction of T2 and ``(vv|vv)`` integrals in CCSD and CCSD-lambda functions.
These techniques allow the CC module to efficiently handle medium-sized
systems.  In a test system with 25 occupied orbitals and 1500 virtual orbitals, each
CCSD iteration takes about 2.5 hours.  The program does not automatically
switch to AO-driven CCSD for large systems.  The user must manually set the
:attr:`direct` attribute to enable an AO-driven CCSD calculation::

    mycc = cc.CCSD(mf)
    mycc.direct = True
    mycc.kernel()

Some of the CC methods have an efficient incore implementation, where
all tensors are held in memory.  The incore implementation reduces the IO overhead
and optimizes certain formulas to gain the best FLOPS.  It is about 30\% faster
than the outcore implementation.  Depending on the available memory, the incore
code can be used for systems with up to approximately 250 orbitals.

Point group symmetry is not considered in the CCSD programs, but it is used in
the CCSD(T) code to gain the best performance.

Arbitrary frozen orbitals (not limited to frozen core) are supported
by the CCSD, CCSD(T), density matrices, and EOM-CCSD modules, but not in
the analytical CCSD gradient module.


.. Add Memory requirements


Examples
========

This section documents some examples about how to effectively use the CCSD
module, and how to incorporate the CCSD solver with other PySCF functions to
perform advanced simulations.  For a complete list of CC examples, see
``pyscf/examples/cc``.

A general solver for customized Hamiltonian
-------------------------------------------
The CC module is not limited to molecular systems.  The program is implemented as
a general solver for arbitrary Hamiltonians.  It allows users to overwrite the
default molecular Hamiltonian with their own effective Hamiltonians.  In this
example, we create a Hubbard model and feed its Hamiltonian to the CCSD module.

.. literalinclude:: ../../examples/cc/40-ccsd_custom_hamiltonian.py


Using CCSD as CASCI active space solver
---------------------------------------
CCSD program can be wrapped as a Full CI solver, which can be combined with the
CASCI solver to approximate the multi-configuration calculation.

.. literalinclude:: ../../examples/cc/42-as_casci_fcisolver.py


Gamma point CCSD with Periodic boundary condition
-------------------------------------------------
Integrals in Gamma point of periodic Hartree-Fock calculation are all real.
You can feed the integrals into any pyscf molecular module using the same
operations as the above example.  However, the interface between PBC code and
molecular code are more compatible.  You can treat the crystal object and the
molecule object in the same manner.  In this example, you can pass the PBC mean
field method to CC module to have the gamma point CCSD correlation.

.. literalinclude:: ../../examples/pbc/12-gamma_point_post_hf.py


CCSD with truncated MOs to avoid linear dependency
--------------------------------------------------
It is common to have linear dependence when one wants to systematically enlarge
the AO basis set to approach complete basis set limit.  The numerical
instability usually has noticeable effects on the CCSD convergence.  An
effective way to remove this negative effects is to truncate the AO sets and
allow the MO orbitals being less than AO functions.

.. literalinclude:: ../../examples/cc/31-remove_linear_dep.py


Response and un-relaxed CCSD density matrix
-------------------------------------------
CCSD has two kinds of one-particle density matrices.  The (second order)
un-relaxed density matrix and the (relaxed) response density matrix.  The
:func:`CCSD.make_rdm1` function computes the un-relaxed density matrix which is
associated to the regular CCSD energy formula.  The response density is mainly
used to compute the first order response quantities eg the analytical nuclear
gradients.  It is not recommended to use the response density matrix for
population analysis.

.. literalinclude:: ../../examples/cc/01-density_matrix.py


Reusing integrals in CCSD and relevant calculations
---------------------------------------------------
By default the CCSD solver and the relevant CCSD lambda solver, CCSD(T), CCSD
gradients program generate MO integrals in their own runtime.  But in most
scenario, the same MO integrals can be generated once and reused in the four
modules.  To remove the overhead of recomputing MO integrals, the three module
support user to feed MO integrals.

.. literalinclude:: ../../examples/cc/12-reuse_integrals.py


Interfering CCSD-DIIS
---------------------


Restart CCSD
------------


Program reference
=================

cc.ccsd module and CCSD class
-----------------------------

The :class:`pyscf.cc.ccsd.CCSD` class is the object to hold the restricted CCSD environment
attributes and results.  The environment attributes are the parameters to
control the runtime behavior of the CCSD module, e.g. the convergence criteria, DIIS
parameters, and so on.  After the ground state CCSD calculation, correlation
energy, ``T1`` and ``T2`` amplitudes are stored in the CCSD object.
This class supports the calculation of CCSD 1- and 2-particle density matrices.

.. autoclass:: pyscf.cc.ccsd.CCSD

.. automodule:: pyscf.cc.ccsd
   :members:

cc.rccsd and RCCSD class
------------------------

:class:`pyscf.cc.rccsd.RCCSD` is also a class for restricted CCSD calculations, but
different to the :class:`pyscf.cc.ccsd.CCSD` class.  It uses different formula
to compute the ground state CCSD solution.  Although slower than the
implmentation in the :class:`pyscf.cc.ccsd.CCSD` class, it supports the system
with complex integrals.  Another difference is that this class supports EOM-CCSD
methods, including EOM-IP-CCSD, EOM-EA-CCSD, EOM-EE-CCSD, EOM-SF-CCSD.

.. autoclass:: pyscf.cc.rccsd.RCCSD

.. automodule:: pyscf.cc.rccsd
   :members:

cc.uccsd and UCCSD class
------------------------

:class:`pyscf.cc.uccsd.UCCSD` class supports the CCSD calculation based on UHF
wavefunction as well as the ROHF wavefunction.  Besides the ground state UCCSD calculation,
UCCSD lambda equation, 1-particle and 2-particle density matrices, EOM-IP-CCSD,
EOM-EA-CCSD, EOM-EE-CCSD are all available in this class.  Note this class does
not support complex integrals.

.. autoclass:: pyscf.cc.uccsd.UCCSD

.. automodule:: pyscf.cc.uccsd
   :members:

cc.addons
---------
Helper functions for CCSD, RCCSD and UCCSD modules are implemented in
:mod:`cc.addons`

.. automodule:: pyscf.cc.addons
   :members:

CCSD(T)
-------
.. Note ``T3`` amplitudes of CCSD(T) is not stored

.. automodule:: pyscf.cc.ccsd_t
   :members:

CCSD gradients
--------------

.. automodule:: pyscf.cc.ccsd_grad
   :members:
