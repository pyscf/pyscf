.. _gto:

gto --- Molecular structure and GTO basis
*****************************************

This module provides the functions to parse the command line options,
the molecular geometry and format the basic functions for `libcint`
integral library.  In :file:`mole`, a basic class :class:`Mole` is
defined to hold the global parameters, which will be used throughout the
package.

Input
=====

Geometry
--------

There are multiple ways to input molecular geometry. The internal format of
:attr:`Mole.atom` is a python list::

  atom = [[atom1, (x, y, z)],
          [atom2, (x, y, z)],
          ...
          [atomN, (x, y, z)]]

You can input the geometry in this format.  You can use Python script to
construct the geometry::

  >>> mol = gto.Mole()
  >>> mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H',(0, 0, 1)]]
  >>> mol.atom.extend([['H', (i, i, i)] for i in range(1,5)])

Besides Python list, tuple and numpy.ndarray are all supported by the internal
format::

  >>> mol.atom = (('O',numpy.zeros(3)), ['H', 0, 1, 0], ['H',[0, 0, 1]])

Also, :attr:`~Mole.atom` can be a string of Cartesian format or Z-matrix format::

  >>> mol = gto.Mole()
  >>> mol.atom = '''
  >>> O 0 0 0
  >>> H 0 1 0
  >>> H 0 0 1;
  >>> '''

There are a few requirements for the string format.  The string input
takes ``;`` or ``\n`` to partition atoms. White space and ``,`` are used to
split items for each atom.  Blank lines or lines started with ``#`` will be
ignored::

  >>> mol = gto.M(
  ... mol.atom = '''
  ... #O 0 0 0
  ... H 0 1 0
  ...
  ... H 0 0 1;
  ... ''')
  >>> mol.natm
  2

The geometry string is case-insensitive.  It also supports to input the nuclear
charges of elements::

  >>> mol = gto.Mole()
  >>> mol.atom = [[8,(0, 0, 0)], ['h',(0, 1, 0)], ['H',(0, 0, 1)]]

If you need to label an atom to distinguish it from the rest, you can prefix
or suffix number or special characters ``1234567890~!@#$%^&*()_+.?:<>[]{}|``
(except ``,`` and ``;``) to an atomic symbol.  With this decoration, you can
specify different basis sets, or masses, or nuclear models for different atoms::

  >>> mol = gto.Mole()
  >>> mol.atom = '''8 0 0 0; h:1 0 1 0; H@2 0 0'''
  >>> mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
  >>> mol.build()
  >>> print(mol.atom)
  [['O', [0.0, 0.0, 0.0]], ['H:1', [0.0, 1.0, 0.0]], ['H@2', [0.0, 0.0]]]

No matter which format or symbols were used in the input, :func:`Mole.build`
will convert :attr:`Mole.atom` to the internal format::

  >>> mol.atom = '''
      O        0,   0, 0             ; 1 0.0 1 0
      
          H@2,0 0 1
      '''
  >>> mol.build()
  >>> print(mol.atom)
  [['O', [0.0, 0.0, 0.0]], ['H', [0.0, 1.0, 0.0]], ['H@2', [0.0, 0.0, 1.0]]]


.. _input_basis:

Input Basis
-----------
There are various ways to input basis sets.  Besides the input of universal
basis string and basis ``dict``::

  mol.basis = 'sto3g'
  mol.basis = {'O': 'sto3g', 'H': '6-31g'}

basis can be input with helper functions.
Function :func:`basis.parse` can parse a basis string of NWChem format
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

Functions :func:`basis.load` can be load arbitrary basis from the database, even
the basis which does not match the element.

  mol.basis = {'H': gto.basis.load('sto3g', 'C')}

Both :func:`basis.parse` and :func:`basis.load` return the basis set in the
internal format (See the :ref:`gto_basis`).

Basis parser supports "Ghost" atom::

  mol.basis = {'GHOST': gto.basis.load('cc-pvdz', 'O'), 'H': 'sto3g'}

More examples of inputing ghost atoms can be found in
:file:`examples/gto/03-ghost_atom.py`

Like the requirements of geometry input, you can use atomic symbol
(case-insensitive) or the atomic nuclear charge, as the keyword of the
:attr:`~Mole.basis` dict.  Prefix and suffix of numbers and special
characters are allowed.  If the decorated atomic symbol is appeared in
:attr:`~Mole.atom` but not :attr:`~Mole.basis`, the basis parser will
remove all decorations then seek the pure atomic symbol in
:attr:`~Mole.basis` dict.  In the following example, ``6-31G`` basis
will be assigned to the second H atom, but ``STO-3G`` will be used for
the third atom::

  mol.atom = '8 0 0 0; h1 0 1 0; H2 0 0 1'
  mol.basis = {'O': 'sto-3g', 'H': 'sto3g', 'H1': '6-31G'}


Command line
------------

Some of the input variables can be passed from command line::

  $ python example.py -o /path/to/my_log.txt -m 1000

This command line specifies the output file and the maximum of memory for the
calculation.  By default, command line has the highest priority, which means our
settings in the script will be overwritten by the command line arguments.  To
make the input parser ignore the command line arguments, you can call the
:func:`Mole.build` with::

  mol.build(0, 0)

The first 0 prevent :func:`~Mole.build` dumping the input file.  The
second 0 prevent :func:`~Mole.build` parsing command line.

 
Program reference
=================

mole
----

:class:`Mole` class handles three layers: input, internal format, libcint arguments.
The relationship of the three layers are::

  .atom (input)  <=>  ._atom (for python) <=> ._atm (for libcint)
  .basis (input) <=> ._basis (for python) <=> ._bas (for libcint)

input layer does not talk to libcint directly.  Data are held in python
internal fomrat layer.  Most of methods defined in this class only operates
on the internal format.  Exceptions are make_env, make_atm_env, make_bas_env,
:func:`set_common_orig_`, :func:`set_rinv_orig_` which are used to
manipulate the libcint arguments.


.. automodule:: pyscf.gto.mole
   :members:

.. autoclass:: Mole
   :members:


.. _gto_moleintor:

moleintor
---------

.. automodule:: pyscf.gto.moleintor
   :members:


.. _gto_basis:

basis
-----

Internal format
^^^^^^^^^^^^^^^

This module loads basis set and ECP data from basis database and parse the basis
(mostly in NWChem format) and finally convert to internal format.  The internal
format of basis set is::

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

For example::

  mol.basis = {'H': [[0,
                      (19.2406000, 0.0328280),
                      (2.8992000, 0.2312080),
                      (0.6534000, 0.8172380),],
                     [0,
                      (0.1776000, 1.0000000),],
                     [1,
                      (1.0000000, 1.0000000),]],
              }

Some basis sets, e.g. :file:`pyscf/gto/basis/dzp_dunning.py`, are saved in the
internal format.

.. automodule:: pyscf.gto.basis
   :members:


