.. _getting_started:


Tutorial
********

Quick setup
===========

The prerequisites of PySCF include `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_,
and `h5py <http://www.h5py.org/>`_.  On the Ubuntu host, you can quickly
install them::

  $ sudo apt-get install python-h5py python-scipy cmake

Then download the latest version of `pyscf <https://github.com/sunqm/pyscf.git/>`_
and build C extensions in :file:`pyscf/lib`::

  $ git clone https://github.com/sunqm/pyscf
  $ cd pyscf/lib
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

Finally, update the Python runtime path :code:`PYTHONPATH` (assuming pyscf
is put in /home/abc, replace it with your own path)::

  $ echo 'export PYTHONPATH=/home/abc:$PYTHONPATH' >> ~/.bashrc
  $ source ~/.bashrc

To ensure the installation is successed, start a Python shell, and type::

  >>> import pyscf

If you got errors like::

  ImportError: No module named pyscf

It's very possible that you put ``/home/abc/pyscf`` in :code:`PYTHONPATH`.
You need to remove the ``/pyscf`` in that string and try
``import pyscf`` in the python shell again.

.. note::  The quick setup will not provide the best performance.
  Please refer to :ref:`installing` for the optimized libraries.


A simple example
================

Here is an example to run HF calculation for hydrogen molecule::

  >>> from pyscf import gto, scf
  >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='cc-pvdz')
  >>> mf = scf.RHF(mol)
  >>> mf.kernel()
  converged SCF energy = -1.06111199785749
  -1.06111199786


Input molecule
==============

The first question is how to input a molecule in pyscf.  There are three
ways to define and build a molecule.  The first is to use the keyword
arguments of :func:`Mole.build` to initialize a molecule::

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

There are two ways to insert the geometry. The internal format of
:attr:`Mole.atom` is a python list::

  atom = [[atom1, (x, y, z)],
          [atom2, (x, y, z)],
          ...
          [atomN, (x, y, z)]]

You can input the geometry in this format.  Therefore, you are able to
use all the possible feature provided by Python to construct the
geometry::

  >>> mol = gto.Mole()
  >>> mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H',(0, 0, 1)]]
  >>> mol.atom.extend([['H', (i, i, i)] for i in range(1,5)])

Although the internal format is defined on the Python list, you can
replace it with tuple, or numpy.ndarray for the coordinate part::

  >>> mol.atom = (('O',numpy.zeros(3)), ['H', 0, 1, 0], ['H',[0, 0, 1]])

The second way is to assign :attr:`~Mole.atom` a string like::

  >>> mol = gto.Mole()
  >>> mol.atom = '''
  >>> O 0 0 0
  >>> H 0 1 0
  >>> H 0 0 1;
  >>> '''
  >>> mol.atom += ';'.join(['H '+(' %f'%i)*3 for i in range(1,5)])

There are a few requirements for the string format.  The string input
takes ``;`` or ``\n`` to partition atoms, and `` `` or ``,`` to divide
the atomic symbol and the coordinates.  Blank lines will be ignored.

.. note:: Z-matrix string is not supported in the present version.

To specify the atoms type, you can use the atomic symbol (case-insensitive),
or the atomic nuclear charge::

  >>> mol = gto.Mole()
  >>> mol.atom = [[8,(0, 0, 0)], ['h',(0, 1, 0)], ['H',(0, 0, 1)]]

If you want to label an atom to distinguish it from the rest, you can prefix
or suffix number or special characters ``1234567890~!@#$%^&*()_+.?:<>[]{}|``
(execept ``,`` and ``;``) to an atomic symbol.  With this decoration, you can
specify a basis set, or mass, or nuclear model for a particular atom
without affect the rest same type atoms::

  >>> mol = gto.Mole()
  >>> mol.atom = '''8 0 0 0; h:1 0 1 0; H@2 0 0'''
  >>> mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
  >>> mol.build()
  >>> print(mol.atom)
  [['O', [0.0, 0.0, 0.0]], ['H:1', [0.0, 1.0, 0.0]], ['H@2', [0.0, 0.0]]]

No matter which format or symbol used for the input, :func:`Mole.build`
will convert :attr:`Mole.atom` to the internal format::

  >>> mol.atom = '''
      O        0,   0, 0             ; 1 0.0 1 0
      
          H@2,0 0 1
      '''
  >>> mol.build()
  >>> print(mol.atom)
  [['O', [0.0, 0.0, 0.0]], ['H', [0.0, 1.0, 0.0]], ['H@2', [0.0, 0.0, 1.0]]]

Basis set
---------

There are four ways to assign basis sets.  One is to input the intenal format::

  basis = {atom_type1:[[angular_momentum
                        (GTO-exp1, contract-coeff11, contract-coeff12),
                        (GTO-exp2, contract-coeff21, contract-coeff22),
                        (GTO-exp3, contract-coeff31, contract-coeff32),
                        ...],
                       [angular_momentum
                        (GTO-exp1, contract-coeff11, contract-coeff12),
                        ...],
                       ...],
           atom_type2:[[angular_momentum, (...),],
                       ...],

like::

  mol.basis = {'H': [[0,
                      (19.2406000, 0.0328280),
                      (2.8992000, 0.2312080),
                      (0.6534000, 0.8172380),],
                     [0,
                      (0.1776000, 1.0000000),],
                     [1,
                      (1.0000000, 1.0000000),]],
              }

You can find more examples of internal format in :file:`pyscf/gto/basis/`.
Some basis sets, e.g.  :file:`dzp_dunning.py`, are saved in the internal
format.

But the internal format is not easy to input.  So two functions
:func:`basis.load` and :func:`basis.parse` are defined to simplify the
workload.  They return the basis set of internal format::

  mol.basis = {'H': gto.basis.load('sto3g', 'H')}

:func:`basis.parse` can parse a basis string of NWChem format
(https://bse.pnl.gov/bse/portal)::

  mol.basis = {'O': gto.basis.parse('''
  C    S
       71.6168370              0.15432897       
       13.0450960              0.53532814       
        3.5305122              0.44463454       
  C    SP
        2.9412494             -0.09996723             0.15591627       
        0.6834831              0.39951283             0.60768372       
        0.2222899              0.70011547             0.39195739       
  ''')}

Things can be more convenient by inputing name of the baiss::

  mol.basis = {'O': 'sto3g', 'H': '6-31g'}

or specify one basis set universally for all atoms::

  mol.basis = '6-31g'

The package defined a 0-nuclear-charge atom, called "GHOST".  This phantom
atom can be used to insert basis for BSSE correction with
:func:`basis.load` and :func:`basis.parse`::

  mol.basis = {'GHOST': gto.basis.load('cc-pvdz', 'O'), 'H': 'sto3g'}

Like the requirements of geometry input, you can use atomic symbol
(case-insensitive) or the atomic nuclear charge, as the keyword of the
:attr:`~Mole.basis` dict.  Prefix and suffix of numbers and special
characters are allowed.  If the decorated atomic symbol is appeared in
:attr:`~Mole.atom` but not :attr:`~Mole.basis`, the basis parser will
remove all decorations then seek the pure atomic symbol in
:attr:`~Mole.basis` dict.  In the following example, ``6-31G`` basis
will be assigned to the second H atom, but ``STO-3G`` will be used for
the third atom::

  mol.atom = [[8,(0, 0, 0)], ['h1',(0, 1, 0)], ['H2',(0, 0, 1)]]
  mol.basis = {'O': 'sto-3g', 'H': 'sto3g', 'H1': '6-31G'}

Other parameters
----------------

You can assign more infomations to a molecular object::

  mol.symmetry = 1
  mol.charge = 1
  mol.spin = 1
  mol.light_speed = 137.035989
  mol.nucmod = {'O1': 1} 
  mol.mass = {'O1': 18, 'H': 2} 

.. note::
  :attr:`Mole.spin` is *2S*, the alpha and beta electron number difference.

:class:`Mole` also defines some global options.  You can control the
print level with :attr:`~Mole.verbose`::

  mol.verbose = 4

The print level can be 0 (quite, no output) to 9 (very noise).  Mostly,
the useful messages are printed at level 4 (info), and 5 (debug).
You can also specify the place where to write the print messages::

  mol.output = 'path/to/my_log.txt'

Without assigning this variable, messages will be printed to
:attr:`sys.stdout`.  You can control the memory usage::

  mol.max_memory = 1000 # MB
  
The default size is set by :attr:`lib.parameters.MEMORY_MAX`.

:attr:`~Mole.verbose`, :attr:`~Mole.output` and :attr:`~Mole.max_memory`
can be assgined from command line::

  $ python example.py --verbose -o /path/to/my_log.txt -m 1000

The command line arguments are parsed in :func:`Mole.build`.  By
default, they have the highest priority, which means our settings in the
script will be overwritten by the command line arguments.  To prevent
that, we can call :func:`Mole.build` with::

  mol.build(0, 0)

The first 0 prevent :func:`~Mole.build` dumping the input file.  The
second 0 prevent :func:`~Mole.build` parsing command line.


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

Import non-relativistic Hartree-Fock::

  >>> from pyscf import scf
  >>> m = scf.RHF(mol)
  >>> print('E(HF) = %g' % m.kernel())
  E(HF) = -149.544214749

But the ground state of oxygen molecule should be triplet::

  >>> o2_tri = mol.copy()
  >>> o2_tri.spin = 2
  >>> o2_tri.build(0, 0)  # two "0"s to prevent dumping input and parsing command line
  >>> rhf3 = scf.RHF(o2_tri)
  >>> print(rhf3.kernel())
  -149.609461122

or run with UHF::

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

Here we rebuild the moleclue because the point group symmetry
information, symmetry adapted orbitals, are initalized in
:meth:`Mole.build`.  With a little more lines of code, we can check the
occupancy for each irreducible representations::

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
function by assigning electron numbers `(alpha,beta)` for particular irreps::

  >>> rhf3_sym.irrep_nelec = {'B2g': (1,1), 'B3g': (1,1), 'B2u': (1,0), 'B3u': (1,0)}
  >>> rhf3_sym.kernel()
  >>> print(rhf3_sym.kernel())
  -148.983117701
  >>> rhf3_sym.get_irrep_nelec()
  {'Ag' : (3, 3), 'B1g': (0, 0), 'B2g': (1, 1), 'B3g': (1, 1), 'Au' : (0, 0), 'B1u': (1, 0), 'B2u': (0, 1), 'B3u': (1, 0)}

More informations can be found in the output file "o2.log".

MP2 and MO integral transformation
----------------------------------

Next, we compute the correlation energy with :mod:`mp.mp2`::

  >>> from pyscf import mp
  >>> mp2 = mp.MP2(m)
  >>> print('E(MP2) = %.9g' % m.kernel()[0])
  E(MP2) = -0.379359288

This is the correlation energy of singlet ground state.  For the triplet
state, we can write our own function to compute the MP2 correlation energy

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

In this example, we concatenate :math:`\alpha` and :math:`\beta`
orbitals to fake the spin-orbitals.  After integral transformation, we
zerod out the integrals of different spin.  Here, the :mod:`ao2mo`
module provides the general 2-electron MO integral transformation.
Using this module, you are able to do *arbitrary* integral
transformation for *arbitrary* integrals. For example, the following
code gives the ``(ov|vv)`` type integrals::

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
  >>> eri = ao2mo.general(mol, (co,cv,co,cv), intor='cint2e_ip1_sph', comp=3)
  >>> eri = eri.reshape(3, nocc, nvir, nocc, nvir)
  >>> print(eri.shape)
  (3, 8, 20, 8, 20)


CASCI and CASSCF
----------------

The two classes :class:`mcscf.CASCI` and :class:`mcscf.CASSCF` provided
by :mod:`mcscf` have the same initialization structure::

  >>> from pyscf import mcscf
  >>> mc = mcscf.CASCI(m, 4, 6)
  >>> print('E(CASCI) = %.9g' % mc.casci()[0])
  E(CASCI) = -149.601051
  >>> mc = mcscf.CASSCF(m, 4, 6)
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.613191

In this example, the CAS space is (6e, 4o): the third argument for
CASCI/CASSCF is the size of CAS space; the fourth argument is the number
of electrons.  By default, the CAS solver splits the electron number
according to the :attr:`Mole.spin` attribute.  In the above example, the
number of alpha electron is equal to the number of beta electrons, since
the ``mol`` object is initialized with ``spin=0``.  The spin
multiplicity of the CASSCF/CASCI solver can be changed by the fourth
argument::

  >>> mc = mcscf.CASSCF(m, 4, (4,2))
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.609461
  >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
  S^2 = 2.0000000, 2S+1 = 3.0000000

The two integers in tuple stand for the number of alpha and beta
electrons.  Although it is a triplet state, the solution might not be
right since the CASSCF is based on the incorrect singlet HF ground
state.  Using the ROHF ground state, we have::

  >>> mc = mcscf.CASSCF(rhf3, 4, 6)
  >>> print('E(CASSCF) = %.9g' % mc.mc1step()[0])
  E(CASSCF) = -149.646746

where we called :func:`mf.mc1step`, which is an alias name of ``mc.kernel``.
The energy is lower than the RHF based wavefunction. Alternatively, we
can also use the UHF ground state to start a CASSCF calculation::

  >>> mc = mcscf.CASSCF(uhf3, 4, 6)
  >>> print('E(CASSCF) = %.9g' % mc.kernel()[0])
  E(CASSCF) = -149.661324
  >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
  S^2 = 3.9713105, 2S+1 = 4.1091656

Woo, the total energy is even lower.  But the spin is contaminated.


Restore previous calculation
============================
There is no `restart` mechanism in this package.  Alternatively,
calculations can be "restored" by proper initial guess.  The initial
guess can be prepared in many ways.  One is to read the ``chkpoint`` file
which is generated in the previous or the other computations::

  >>> from pyscf import scf
  >>> mf = scf.RHF(mol)
  >>> mf.chkfile = '/path/to/chkfile'
  >>> mf.init_guess = 'chkfile'
  >>> mf.kernel()

``/path/to/chkfile`` can be found in the output of the other calculation
(if mol.verbose >= 4, there is an entry "chkfile to save SCF result" to
record the name of chkfile in the output).  By setting
:attr:`chkfile` and :attr:`init_guess`, the SCF module can read the
molecular orbitals stored in the given :attr:`chkfile` and rotate them to
the proper basis sets.  There is another way to read the initial guess::

  >>> from pyscf import scf
  >>> mf = scf.RHF(mol)
  >>> dm = scf.hf.from_chk(mol, '/path/to/chkfile')
  >>> mf.kernel(dm)

:func:`scf.hf.from_chk` reads the chkpoint file and generated the
corresponding density matrix.  The density matrix is then
feed to :func:`mf.kernel`, which takes one parameter as the start point
for SCF loops.

The "chkfile" is not limited to the calculation based on the same
molecular and same basis set.  One can also first do a cheap SCF (with
small basis sets) or a model SCF (dropping a few atoms, or charged
system), then use :func:`scf.hf.from_chk` to project the
results to the target basis sets.  :mod:`scf` provides other initial
guess methods such as :func:`scf.hf.init_guess_by_minao`,
:func:`scf.hf.init_guess_by_atom`, :func:`scf.hf.init_guess_by_1e` (you
can use :func:`scf.hf.get_init_guess` function to call them).  If you
like, you can mix all kinds of methods to make a initial guess for
density matrix and feed it to :func:`mf.scf`

To restore CASSCF calculation, you need prepare either CASSCF orbitals
or CI coefficients (not that useful unless doing DMRG-CASSCF) or both.
For instance, see ``pyscf/examples/mcscf/13-restart.py``

.. literalinclude:: ../../examples/mcscf/13-restart.py

Access AO integrals
===================

Libcint interface
-----------------

Pyscf uses `Libcint <https://github.com/sunqm/libcint>`_ library as the AO
integral backend.  It provides simple interface functon :func:`getints_by_shell`
to call the functions provided by ``Libcint``.  Now let's try to access
3-center 2-electron integrals through the interface::

  import numpy
  from pyscf import gto, scf, df
  mol = gto.M(atom='O 0 0 0; h 0 -0.757 0.587; h 0 0.757 0.587', basis='cc-pvdz')
  auxmol = gto.M(atom='O 0 0 0; h 0 -0.757 0.587; h 0 0.757 0.587', basis='weigend')
  atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
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
              buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
              di, dj, dk = buf.shape
              eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
              pk += dk
          pj += dj
      pi += di

Here we first read in the Weigend density fitting basis to ``auxmol``.  In
libcint, all integral functions requires the same input arguments,
``(buf, shells, atm, bas, env, opt)``.  So we use :func:`gto.conc_env`
to concatenate the AO basis and the auxiliary fitting basis, and
obtain one set of input arguments ``atm, bas, env``.  In the resultant
``bas``, the first ``mol.nbas`` entries store the AO basis, and the following
``auxmol.nbas`` are auxiliary basis.  Then the three nested loops run over all
integrals for the three index integral `(ij|K)`.  Next, we compute the two
center integrals::

  eri2c = numpy.empty((naux,naux))
  pk = 0
  for k in range(mol.nbas, mol.nbas+auxmol.nbas):
      pl = 0
      for l in range(mol.nbas, mol.nbas+auxmol.nbas):
          shls = (k, l)
          buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
          dk, dl = buf.shape
          eri2c[pk:pk+dk,pl:pl+dl] = buf
          pl += dl
      pk += dk

Yes, we are ready to implement our own density fitting Hartree-Fock now!

.. code:: python

  def get_vhf(mol, dm, *args, **kwargs):
      naux = eri2c.shape[0]
      rho = numpy.einsum('ijp,ij->p', eri3c, dm)
      rho = numpy.linalg.solve(eri2c, rho)
      jmat = numpy.einsum('p,ijp->ij', rho, eri3c)
      kpj = numpy.einsum('ijp,jk->ikp', eri3c, dm)
      pik = numpy.linalg.solve(eri2c, kpj.reshape(-1,naux).T)
      kmat = numpy.einsum('pik,kjp->ij', pik, eri3c)
      return jmat - kmat * .5
      
  mf = scf.RHF(mol)
  mf.verbose = 0
  mf.get_veff = get_vhf
  print('E(DF-HF) = %.12f, ref = %.12f' % (mf.kernel(), scf.density_fit(mf).kernel()))

Hopefully, your screen will print out

  | E(DF-HF) = -76.025936299702, ref = -76.025936299702

as mine.


One electron AO integrals
-------------------------

There many ways to get the one-electron integrals from pyscf.  Apparently, one
is to call :func:`getints_by_shell` like the previous example.  The other
method is to call :func:`got.getints`::

  >>> from pyscf import gto
  >>> mol = gto.M(atom='h 0 0 0; f 0 0 1', basis='sto-3g')
  >>> hcore = gto.getints('cint1e_nuc_sph', mol.atm, mol.bas, mol.env) + gto.getints('cint1e_kin_sph', mol.atm, mol.bas, mol.env)
  >>> ovlp = gto.getints('cint1e_ovlp_sph', mol.atm, mol.bas, mol.env)

Actually, there is an even simpler function :class:`Mole.intor`::

  >>> hcore = mol.intor('cint1e_nuc_sph') + mol.intor('cint1e_kin_sph')
  >>> ovlp = mol.intor('cint1e_ovlp_sph')

There is a long list of supported AO integrals.  See :ref:`gto_moleintor`.


More examples
=============

Hartree-Fock with density fitting

.. literalinclude:: ../../examples/scf/20-density_fitting.py

Use :mod:`scf.hf.SCF` to simulate Hubbard model

.. literalinclude:: ../../examples/scf/40-hf_with_given_hamiltonian.py

Symmetry in CASSCF

.. literalinclude:: ../../examples/mcscf/21-nosymhf_then_symcasscf.py

