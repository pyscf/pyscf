.. _cc:

:mod:`cc` --- Coupled Cluster
*****************************

.. module:: cc
   :synopsis: Computing Coulpled Cluster energy and properties
.. sectionauthor:: Qiming Sun <osirpt.sun@gmail.com>.

The :mod:`cc` module implements the coupled cluster (CC) model to compute
energy, analytical nuclear gradients, density matrices, excitation states and
relevant properties.

To compute CC energy, you need first create the mean-field calculation using
the mean-field module :mod:`scf`.  The mean-field object defines the Hamiltonian
and the problem size which are used to initialize the CC object::

    from pyscf import gto, scf, cc
    mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    mycc.kernel()

Unrelaxed density matrices are evaluated in MO basis::

    dm1 = mycc.make_rdm1()
    dm2 = mycc.make_rdm2()

CCSD(T) energy can be obtained by::

    from pyscf.cc import ccsd_t
    print(ccsd_t.kernel(mycc, mycc.ao2mo())[0])

Gradients are avaialbe::

    from pyscf.cc import ccsd_grad
    from pyscf import grad
    grad_e = ccsd_grad.kernel(mycc)
    grad_n = grad.grad_nuc(mol)
    grad = grad_e + grad_nuc

The IP/EA-EOM-CCSD can be calculated::

    mycc = cc.RCCSD(mf)
    mycc.kernel()
    eip,cip = mycc.ipccsd(nroots=1)
    eee,cee = mycc.eaccsd(nroots=1)

    mycc = cc.UCCSD(mf)
    mycc.kernel()
    eip,cip = mycc.ipccsd(nroots=1)
    eea,cea = mycc.eeccsd(nroots=1)


All CC methods have two implementations.  One is simple for readablity (suffixed
by ``_slow`` in the filename) and the other is extensively optimized for
computing efficiency.  In the slow version,  :func:`numpy.einsum` function is
used as the tensor contraction engine.  The code is structured as close as
possible to the formula documented in the literature.  Pure Python/numpy
data structure and functions are used so that the memory management is avoided.
It is easy to make modification or functionalize new methods based on the slow
implementations.

The computing efficient (outcore version) is the default implementation for CC
module.  In this implementation, the CPU usage, memory footprint, memory
efficiency, and IO overhead are carefully considered.  To keep small memory
footprint, most integral tensors are stored on disk.  IO is one of the main
bottleneck in this implementation.  Two techniques have be used to reduce the IO
overhead.  One is the asynchronized IO to overlap the computation and
reading/writing of the 4-indices tensor.  The other is AO-driven for the
contraction of T2 and ``(vv|vv)`` integrals in CCSD and CCSD-lambda functions.
These techniques allows the CC module to efficiently handle medium size systems.
In a test system which has 25 occupied, 1500 virtual orbitals, each CCSD
iteration takes about 2.5 hours.  The program does not automatically switch to
AO-driven CCSD for large systems.  You need manually set the :attr:`direct` to
enable the AO-driven CCSD calculation::

    mycc = cc.CCSD(mf)
    mycc.direct = True
    mycc.kernel()

Some of the CC methods has the more efficient incore implementation which have
all tensors held in memory.  The incore implementation reduces the IO overhead
and optimizes certain formula to gain the best FLOPS.  It is about 30\% faster
than the outcore implementation.  Depending on the memory size, the roof of
incore code is around 250 orbitals.

Point group symmetry is not considered in the CCSD program.  But it is used in
the CCSD(T) code to gain the best performance.

Arbitrary orbital frozen, which is not limited to the frozen core, is supported
by the CCSD, CCSD(T), density matrices, EOM-CCSD modules.  But not considered in
the analytical CCSD gradients.


Data structure
==============

The :class:`CCSD` class is the object to hold the restricted CCSD environment
attributes and results.  The environment attributes are the parameters to
control the runtime behaviour of CCSD module, eg the convergence criteria, DIIS
parameters.

.. autoclass:: pyscf.cc.ccsd.CCSD

CCSD 1 and 2-particle density matrices, ``T3`` amplitudes of CCSD(T) are not stored. 


Examples
========

This section documents some examples about how to effectively use CCSD
solver and incorporate CCSD solver with other PySCF functions to do advanced
simulations.  For a complete list of CC examples, please see
``pyscf/examples/cc``.

A general solver for given Hamiltonian
--------------------------------------
The CC module is not limited to molecule system.  The program is implemented as
a general solver for arbitrary Hamiltonians.  It allows users overwriting the
default molecule Hamiltonian with their own effective Hamiltonians.  In this
example, we'll create a Hubbard model, and feed its Hamiltonian to CCSD program.

.. literalinclude:: ../../examples/cc/40-ccsd_with_given_hamiltonian.py


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


APIs
====

CCSD
----

.. automodule:: pyscf.cc.ccsd
   :members:

.. automodule:: pyscf.cc.addons
   :members:

CCSD(T)
-------

.. automodule:: pyscf.cc.ccsd_t
   :members:

UCCSD
-----

.. automodule:: pyscf.cc.uccsd
   :members:

CCSD gradients
--------------

.. automodule:: pyscf.cc.ccsd_grad
   :members:

