.. _getting_started:


Tutorial
********

This tutorial shows how to use PySCF package in the perspective of method
development. It involves some knowledge of coding in Python.  An Ipython
notebook of user-guide can be found in
https://github.com/nmardirossian/PySCF_Tutorial.

Quick setup
===========

You can install PySCF from github repo::

  $ sudo apt-get install python-h5py python-scipy cmake
  $ git clone https://github.com/sunqm/pyscf
  $ cd pyscf/pyscf/lib
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

You may need to update the Python runtime searching path :code:`PYTHONPATH`
(assuming the pyscf source code is put in /home/abc, replacing it with your own
path)::

  $ echo 'export PYTHONPATH=/home/abc/pyscf:$PYTHONPATH' >> ~/.bashrc
  $ source ~/.bashrc

.. note::  The quick setup does not provide the best performance.
  Please see :ref:`installing` for the installation with optimized libraries.


A simple example
================

Here is an example to run HF calculation for hydrogen molecule::

  >>> from pyscf import gto, scf
  >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
  >>> mf = scf.RHF(mol)
  >>> mf.kernel()
  converged SCF energy = -1.06111199785749
  -1.06111199786


Initializing a molecule
=======================

There are three ways to define and initialize a molecule.  The first is to use
the keyword arguments of :func:`Mole.build` to initialize a molecule::

  >>> from pyscf import gto
  >>> mol = gto.Mole()
  >>> mol.build(
  ...     atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
  ...     basis = 'sto-3g')

The second way is to assign the geometry, basis etc. to :class:`Mole`
object, then call :meth:`~Mole.build` function to initialize the
molecule::

  >>> mol = gto.Mole()
  >>> mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
  >>> mol.basis = 'sto-3g'
  >>> mol.build()

The third way is to use the shortcut function :func:`Mole.M`.  This
function pass all arguments to :func:`Mole.build`::

  >>> from pyscf import gto
  >>> mol = gto.M(
  ...     atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
  ...     basis = 'sto-3g')

Either way, you may have noticed two keywords ``atom`` and ``basis``.
They are used to hold the molecular geometry and basis sets.

Geometry
--------

Molecular geometry can be input in Cartesian format::

  >>> mol = gto.Mole()
  >>> mol.atom = '''O 0, 0, 0
  ... H   0  1  0; H 0, 0, 1'''

The atoms in the molecule are represented by an element symbol plus three
numbers for coordinates.  Different atoms should be separated by ``;`` or line
break. In the same atom, ``,`` can be used to separate different items.
Z-matrix input format is also supported by the input parser::

  >>> mol = gto.Mole()
  >>> mol.atom = '''O
  ... H, 1, 1.2;  H   1 1.2   2 105'''

Similarly,  different atoms need to be separated by ``;`` or line break.
If you need to label an atom to distinguish it from the rest, you can prefix
or suffix number or special characters ``1234567890~!@#$%^&*()_+.?:<>[]{}|``
(except ``,`` and ``;``) to an atomic symbol.  With this decoration, you can
specify different basis sets, or masses, or nuclear models for different atoms::

  >>> mol = gto.Mole()
  >>> mol.atom = '''8 0 0 0; h:1 0 1 0; H@2 0 0'''
  >>> mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
  >>> mol.build()
  >>> print(mol._atom)
  [['O', [0.0, 0.0, 0.0]], ['H:1', [0.0, 1.0, 0.0]], ['H@2', [0.0, 0.0]]]

Basis set
---------

The simplest way is to assign a string of basis name to :attr:`mol.basis`::

  mol.basis = 'sto3g'

This input will apply the specified basis set to all atoms.  The basis name in
the string is case insensitive.  White space, dash and underscore in the basis
name are all ignored.  If different basis sets are required for different
elements,  a python ``dict`` can be assigned to the basis attribute::

  mol.basis = {'O': 'sto3g', 'H': '6-31g'}

You can find more examples in section :ref:`input_basis` and in the file
:file:`examples/gto/04-input_basis.py`.

Other parameters
----------------

You can assign more informations to the molecular object::

  mol.symmetry = 1
  mol.charge = 1
  mol.spin = 1
  mol.nucmod = {'O1': 1} 
  mol.mass = {'O1': 18, 'H': 2} 

.. note::
  :attr:`Mole.spin` is *2S*, the unpaired electrons = the difference between the
  numbers of alpha and beta electrons.

:class:`Mole` also defines some global parameters.  You can control the
print level globally with :attr:`~Mole.verbose`::

  mol.verbose = 4

The print level can be 0 (quite, no output) to 9 (very noise).  Mostly,
the useful messages are printed at level 4 (info), and 5 (debug).
You can also specify the place where to write the output messages::

  mol.output = 'path/to/my_log.txt'

Without assigning this variable, messages will be dumped to :attr:`sys.stdout`.
You can control the maximum memory usage globally::

  mol.max_memory = 1000 # MB
  
The default size can be defined with shell environment variable `PYSCF_MAX_MEMORY`

:attr:`~Mole.output` and :attr:`~Mole.max_memory` can be assigned from command
line::

  $ python example.py -o /path/to/my_log.txt -m 1000


Initializing a crystal
======================

Initialization a crystal unit cell is very similar to the initialization
molecular object.  Here, :class:`pyscf.pbc.gto.Cell` class should be used
instead of the :class:`pyscf.gto.Mole` class::

  >>> from pyscf.pbc import gto
  >>> cell = gto.Cell()
  >>> cell.atom = '''H  0 0 0; H 1 1 1'''
  >>> cell.basis = 'gth-dzvp'
  >>> cell.pseudo = 'gth-pade'
  >>> cell.a = numpy.eye(3) * 2
  >>> cell.build()

The crystal initialization requires an extra parameter :attr:`cell.a` which
represents the lattice vectors. In the above example, we specified
:attr:`cell.pseudo` for the pseudo-potential of the system which is an optional
parameter.  The input format of basis set is the same to that of :class:`Mole`
object.  The other attributes of :class:`Mole` object such as :attr:`verbose`,
:attr:`max_memory`, :attr:`spin` can also be used in the crystal systems.
More details of the crystal :class:`Cell` object and the relevant input
parameters are documented in :ref:`pbc_gto`.

1D and 2D systems
-----------------

PySCF PBC module supports the low-dimensional PBC systems.  You can initialize
the attribute :attr:`cell.dimension` to specify the dimension of the system::

  >>> from pyscf.pbc import gto
  >>> cell = gto.Cell()
  >>> cell.atom = '''H  0 0 0; H 1 1 0'''
  >>> cell.basis = 'sto3g'
  >>> cell.dimension = 2
  >>> cell.a = numpy.eye(3) * 2
  >>> cell.build()

When :attr:`cell.dimension` is specified, a vacuum of infinite size will be
applied on certain dimension(s).  More specifically, when :attr:`cell.dimension`
is 2, the z-direction will be treated as infinite large and the xy-plane
constitutes the periodic surface. When :attr:`cell.dimension` is 1, y and z axes
are treated as vacuum thus wire is placed on the x axis.  When
:attr:`cell.dimension` is 0, all three directions are vacuum.  The PBC system is
actually the same to the molecular system.

HF, MP2, MCSCF
==============

Hartree-Fock
------------

Now we are ready to study electronic structure theory with pyscf.  Let's
take oxygen molecule as the first example::

  >>> from pyscf import gto
  >>> mol = gto.Mole()
  >>> mol.verbose = 5
  >>> mol.output = 'o2.log'
  >>> mol.atom = 'O 0 0 0; O 0 0 1.2'
  >>> mol.basis = 'ccpvdz'
  >>> mol.build()

Apply non-relativistic Hartree-Fock::

  >>> from pyscf import scf
  >>> m = scf.RHF(mol)
  >>> print('E(HF) = %g' % m.kernel())
  E(HF) = -149.544214749

The ground state of oxygen molecule should be triplet.  So we change the spin to
``2`` (2 more alpha electrons than beta electrons)::

  >>> o2_tri = mol.copy()
  >>> o2_tri.spin = 2
  >>> o2_tri.build(0, 0)  # two "0"s to prevent dumping input and parsing command line
  >>> rhf3 = scf.RHF(o2_tri)
  >>> print(rhf3.kernel())
  -149.609461122

Run UHF::

  >>> uhf3 = scf.UHF(o2_tri)
  >>> print(uhf3.scf())
  -149.628992314
  >>> print('S^2 = %f, 2S+1 = %f' % uhf3.spin_square())
  S^2 = 2.032647, 2S+1 = 3.021686

where we called :func:`mf.scf`, which is an alias name of ``mf.kernel``.
You can impose symmetry::

  >>> o2_sym = mol.copy()
  >>> o2_sym.spin = 2
  >>> o2_sym.symmetry = 1
  >>> o2_sym.build(0, 0)
  >>> rhf3_sym = scf.RHF(o2_sym)
  >>> print(rhf3_sym.kernel())
  -149.609461122

Here we rebuild the molecule because we need to initialize the point group
symmetry information, symmetry adapted orbitals.  We can check the occupancy for
each irreducible representations::

  >>> import numpy
  >>> from pyscf import symm
  >>> def myocc(mf):
  ...     mol = mf.mol
  ...     irrep_id = mol.irrep_id
  ...     so = mol.symm_orb
  ...     orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
  ...     doccsym = numpy.array(orbsym)[mf.mo_occ==2]
  ...     soccsym = numpy.array(orbsym)[mf.mo_occ==1]
  ...     for ir,irname in enumerate(mol.irrep_name):
  ...         print('%s, double-occ = %d, single-occ = %d' %
  ...               (irname, sum(doccsym==ir), sum(soccsym==ir)))
  >>> myocc(rhf3_sym)
  Ag, double-occ = 3, single-occ = 0
  B1g, double-occ = 0, single-occ = 0
  B2g, double-occ = 0, single-occ = 1
  B3g, double-occ = 0, single-occ = 1
  Au, double-occ = 0, single-occ = 0
  B1u, double-occ = 2, single-occ = 0
  B2u, double-occ = 1, single-occ = 0
  B3u, double-occ = 1, single-occ = 0

To label the irreducible representation of given orbitals,
:func:`symm.label_orb_symm` needs the information of the point group
symmetry which are initialized in ``mol`` object, including the `id` of
irreducible representations :attr:`Mole.irrep_id` and the symmetry
adapted basis :attr:`Mole.symm_orb`.  For each :attr:`~Mole.irrep_id`,
:attr:`Mole.irrep_name` gives the associated irrep symbol (A1, B1 ...).
In the SCF calculation, you can control the symmetry of the wave
function by assigning the number of alpha electrons and beta electrons
`(alpha,beta)` for some irreps::

  >>> rhf3_sym.irrep_nelec = {'B2g': (1,1), 'B3g': (1,1), 'B2u': (1,0), 'B3u': (1,0)}
  >>> rhf3_sym.kernel()
  >>> print(rhf3_sym.kernel())
  -148.983117701
  >>> rhf3_sym.get_irrep_nelec()
  {'Ag' : (3, 3), 'B1g': (0, 0), 'B2g': (1, 1), 'B3g': (1, 1), 'Au' : (0, 0), 'B1u': (1, 0), 'B2u': (0, 1), 'B3u': (1, 0)}

More informations of the calculation can be found in the output file ``o2.log``.


MP2 and MO integral transformation
----------------------------------

Next, we compute the correlation energy with :mod:`mp.mp2`::

  >>> from pyscf import mp
  >>> mp2 = mp.MP2(m)
  >>> print('E(MP2) = %.9g' % mp2.kernel()[0])
  E(MP2) = -0.379359288

This is the correlation energy of singlet ground state.  For the triplet
state, we can write a function to compute the correlation energy

.. math::

  E_{corr} = \frac{1}{4}\sum_{ijab}
           \frac{\langle ij||ab \rangle \langle ab||ij \rangle}
           {\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}

.. code:: python

  def myump2(mf):
      import numpy
      from pyscf import ao2mo
      # As UHF objects, mo_energy, mo_occ, mo_coeff are two-item lists
      # (the first item for alpha spin, the second for beta spin).
      mo_energy = mf.mo_energy
      mo_occ = mf.mo_occ
      mo_coeff = mf.mo_coeff
      o = numpy.hstack((mo_coeff[0][:,mo_occ[0]>0] ,mo_coeff[1][:,mo_occ[1]>0]))
      v = numpy.hstack((mo_coeff[0][:,mo_occ[0]==0],mo_coeff[1][:,mo_occ[1]==0]))
      eo = numpy.hstack((mo_energy[0][mo_occ[0]>0] ,mo_energy[1][mo_occ[1]>0]))
      ev = numpy.hstack((mo_energy[0][mo_occ[0]==0],mo_energy[1][mo_occ[1]==0]))
      no = o.shape[1]
      nv = v.shape[1]
      noa = sum(mo_occ[0]>0)
      nva = sum(mo_occ[0]==0)
      eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)
      eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
      g = eri - eri.transpose(0,3,2,1)
      eov = eo.reshape(-1,1) - ev.reshape(-1)
      de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
      emp2 = .25 * numpy.einsum('iajb,iajb,iajb->', g, g, de)
      return emp2

.. code:: python

  >>> print('E(UMP2) = %.9g' % myump2(uhf3))
  -0.346926068

In this example, we concatenate :math:`\alpha` and :math:`\beta` orbitals to
mimic the spin-orbitals.  After integral transformation, we zeroed out the
integrals of different spin.  Here, the :mod:`ao2mo` module provides the general
2-electron MO integral transformation.  Using this module, you are able to do
*arbitrary* integral transformation for *arbitrary* integrals. For example, the
following code gives the ``(ov|vv)`` type integrals::

  >>> from pyscf import ao2mo
  >>> import h5py
  >>> mocc = m.mo_coeff[:,m.mo_occ>0]
  >>> mvir = m.mo_coeff[:,m.mo_occ==0]
  >>> ao2mo.general(mol, (mocc,mvir,mvir,mvir), 'tmp.h5', compact=False)
  >>> feri = h5py.File('tmp.h5')
  >>> ovvv = numpy.array(feri['eri_mo'])
  >>> print(ovvv.shape)
  (160, 400)

We pass ``compact=False`` to :func:`ao2mo.general` to prevent the
function using the permutation symmetry between the virtual-virtual pair
of ``|vv)``.  So the shape of ``ovvv`` corresponds to 8 occupied
orbitals by 20 virtual orbitals for electron 1 ``(ov|`` and 20 by 20 for
electron 2 ``|vv)``.  In the following example, we transformed the
analytical gradients of 2-electron integrals

.. math::

  \langle (\frac{\partial}{\partial R} \varphi_i) \varphi_k | \varphi_j \varphi_l \rangle
  = \int \frac{\frac{\partial\varphi_i(r_1)}{\partial R}
  \varphi_j(r_1) \varphi_k(r_2)\varphi_l(r_2)}{|r_1-r_2|} dr_1 dr_2

.. code:: python

  >>> nocc = mol.nelectron // 2
  >>> co = mf.mo_coeff[:,:nocc]
  >>> cv = mf.mo_coeff[:,nocc:]
  >>> nvir = cv.shape[1]
  >>> eri = ao2mo.general(mol, (co,cv,co,cv), intor='int2e_ip1_sph', comp=3)
  >>> eri = eri.reshape(3, nocc, nvir, nocc, nvir)
  >>> print(eri.shape)
  (3, 8, 20, 8, 20)


CASCI and CASSCF
----------------

The two classes :class:`mcscf.CASCI` and :class:`mcscf.CASSCF` provided
by :mod:`mcscf` have the same initialization interface::

  >>> from pyscf import mcscf
  >>> mc = mcscf.CASCI(m, 4, 6)
  >>> print('E(CASCI) = %.9g' % mc.casci()[0])
  E(CASCI) = -149.601051
  >>> mc = mcscf.CASSCF(m, 4, 6)
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.613191

In this example, the CAS space is (6e, 4o): the third argument for
CASCI/CASSCF is the size of CAS space; the fourth argument is the number
of electrons.  By default, the CAS solver determines the alpha-electron number
and beta-electron number based on the attribute :attr:`Mole.spin`.  In the
above example, the number of alpha electrons is equal to the number of beta
electrons, since the ``mol`` object is initialized with ``spin=0``.  The spin
multiplicity of the CASSCF/CASCI solver can be changed by the fourth argument::

  >>> mc = mcscf.CASSCF(m, 4, (4,2))
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.609461
  >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
  S^2 = 2.0000000, 2S+1 = 3.0000000

The two integers in the tuple represent the number of alpha and beta electrons.
Although it is a triplet state, the solution might not be correct since the
CASSCF is based on the incorrect singlet HF ground state.  Starting from the
ROHF ground state, we have::

  >>> mc = mcscf.CASSCF(rhf3, 4, 6)
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.646746

The energy is lower than the RHF initial guess.
.. We can also use the UHF ground
.. state to start a CASSCF calculation::
.. 
..   >>> mc = mcscf.CASSCF(uhf3, 4, 6)
..   >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
..   E(CASSCF) = -149.661324
..   >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
..   S^2 = 3.9713105, 2S+1 = 4.1091656
.. 
.. Woo, the total energy is even lower.  But the spin is contaminated.


Restore an old calculation
==========================
There is no `restart` mechanism available in PySCF package.  Calculations can be
"restarted" by the proper initial guess.  For SCF, the initial guess can be
prepared in many ways.  One is to read the ``chkpoint`` file
which is generated in the previous or other calculations::

  >>> from pyscf import scf
  >>> mf = scf.RHF(mol)
  >>> mf.chkfile = '/path/to/chkfile'
  >>> mf.init_guess = 'chkfile'
  >>> mf.kernel()

``/path/to/chkfile`` can be found in the output in the calculation
(if mol.verbose >= 4, the filename of the chkfile will be dumped in the output).
By setting :attr:`chkfile` and :attr:`init_guess`, the SCF module can read the
molecular orbitals from the given :attr:`chkfile` and rotate them to
representation of the required basis.  The example
:file:`examples/scf/15-initial_guess.py` records other methods to generate SCF
initial guess.

Initial guess can be fed to the calculation directly.  For example, we can read
the initial guess form a chkfile and achieve the same effects as the on in the
previous example::

  >>> from pyscf import scf
  >>> mf = scf.RHF(mol)
  >>> dm = scf.hf.from_chk(mol, '/path/to/chkfile')
  >>> mf.kernel(dm)

:func:`scf.hf.from_chk` reads the chkpoint file and generates the corresponding
density matrix represented in the required basis.

Initial guess ``chkfile`` is not limited to the calculation based on the same
molecular and same basis set.  One can first do a cheap SCF (with
small basis sets) or a model SCF (dropping a few atoms, or charged
system), then use :func:`scf.hf.from_chk` to project the
results to the target basis sets.

To restart a CASSCF calculation, you need prepare either CASSCF orbitals
or CI coefficients (not that useful unless doing a DMRG-CASSCF calculation) or
both.  For example:

.. literalinclude:: ../../examples/mcscf/13-restart.py

Access AO integrals
===================

molecular integrals
-------------------

PySCF uses `Libcint <https://github.com/sunqm/libcint>`_ library as the AO
integral engine.  It provides simple interface function :func:`getints_by_shell`
to evaluate integrals.  The following example evaluates 3-center 2-electron
integrals with this function::

  import numpy
  from pyscf import gto, scf, df
  mol = gto.M(atom='O 0 0 0; h 0 -0.757 0.587; h 0 0.757 0.587', basis='cc-pvdz')
  auxmol = gto.M(atom='O 0 0 0; h 0 -0.757 0.587; h 0 0.757 0.587', basis='weigend')
  pmol = mol + auxmol
  nao = mol.nao_nr()
  naux = auxmol.nao_nr()
  eri3c = numpy.empty((nao,nao,naux))
  pi = 0
  for i in range(mol.nbas):
      pj = 0
      for j in range(mol.nbas):
          pk = 0
          for k in range(mol.nbas, mol.nbas+auxmol.nbas):
              shls = (i, j, k)
              buf = pmol.intor_by_shell('int3c2e_sph', shls)
              di, dj, dk = buf.shape
              eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
              pk += dk
          pj += dj
      pi += di

Here we load the Weigend density fitting basis to ``auxmol`` and append the
basis to normal orbital basis which was initialized in ``mol``.  In the result
``pmol`` object, the first ``mol.nbas`` shells are the orbital basis and
the next ``auxmol.nbas`` are auxiliary basis.  The three nested loops run over
all integrals for the three index integral `(ij|K)`.  Similarly, we can compute
the two center Coulomb integrals::

  eri2c = numpy.empty((naux,naux))
  pk = 0
  for k in range(mol.nbas, mol.nbas+auxmol.nbas):
      pl = 0
      for l in range(mol.nbas, mol.nbas+auxmol.nbas):
          shls = (k, l)
          buf = pmol.intor_by_shell('int2c2e_sph', shls)
          dk, dl = buf.shape
          eri2c[pk:pk+dk,pl:pl+dl] = buf
          pl += dl
      pk += dk

Now we can use the two-center integrals and three-center integrals to implement
the density fitting Hartree-Fock code.

.. code:: python

  def get_vhf(mol, dm, *args, **kwargs):
      naux = eri2c.shape[0]
      nao = mol.nao_nr()
      rho = numpy.einsum('ijp,ij->p', eri3c, dm)
      rho = numpy.linalg.solve(eri2c, rho)
      jmat = numpy.einsum('p,ijp->ij', rho, eri3c)
      kpj = numpy.einsum('ijp,jk->ikp', eri3c, dm)
      pik = numpy.linalg.solve(eri2c, kpj.reshape(-1,naux).T)
      kmat = numpy.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), eri3c)
      return jmat - kmat * .5
      
  mf = scf.RHF(mol)
  mf.verbose = 0
  mf.get_veff = get_vhf
  print('E(DF-HF) = %.12f, ref = %.12f' % (mf.kernel(), scf.density_fit(mf).kernel()))

Your screen should output

  | E(DF-HF) = -76.025936299702, ref = -76.025936299702


Evaluating the integrals with nested loops and :func:`mol.intor_by_shell` method is
inefficient.  It is preferred to load integrals in bulk and this can be done
with :func:`mol.intor` method::

  eri2c = auxmol.intor('int2c2e_sph')
  eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
  eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

:func:`mol.intor` method can be used to evaluate one-electron integrals,
two-electron integrals::

  hcore = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
  overlap = mol.intor('int1e_ovlp_sph')
  eri = mol.intor('int2e_sph')

There is a long list of supported AO integrals.  See :ref:`gto_moleintor`.


PBC AO integrals
----------------

:func:`mol.intor` can only be used to evaluate the integrals with open boundary
conditions.  When the periodic boundary conditions of crystal systems are
studied, you need to use :func:`pbc.Cell.pbc_intor` function to evaluate the
integrals of short-range operators, such as the overlap, kinetic matrix::

  from pyscf.pbc import gto
  cell = gto.Cell()
  cell.atom = 'H 0 0 0; H 1 1 1'
  cell.a = numpy.eye(3) * 2.
  cell.build()
  overlap = cell.pbc_intor('int1e_ovlp_sph')

By default, :func:`pbc.Cell.pbc_intor` function returns the :math:`\Gamma`-point
integrals.  If k-points are specified, function :func:`pbc.Cell.pbc_intor` can
also evaluate the k-point integrals::

  kpts = cell.make_kpts([2,2,2])  # 8 k-points
  overlap = cell.pbc_intor('int1e_ovlp_sph', kpts=kpts)

.. note:: :func:`pbc.Cell.pbc_intor` can only be used to evaluate the short-range
  integrals.  PBC density fitting method has to be used to compute the
  long-range operator such as nuclear attraction integrals, Coulomb integrals.

The two-electron Coulomb integrals can be evaluated with PBC density fitting
methods::

    from pyscf.pbc import df
    eri = df.DF(cell).get_eri()

See also :ref:`pbc_df` for more details of the PBC density fitting module.


Other features
==============
Density fitting
---------------

.. literalinclude:: ../../examples/scf/20-density_fitting.py

Customizing Hamiltonian
-----------------------

.. literalinclude:: ../../examples/scf/40-customizing_hamiltonian.py

Symmetry in CASSCF
------------------

.. literalinclude:: ../../examples/mcscf/21-nosymhf_then_symcasscf.py

