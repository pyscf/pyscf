.. _advanced:


Advanced topics
***************

Symmetry
========
PySCF supports D2h symmetry and linear molecule symmetry (Dooh and
Coov).  For D2h, the direct production of representations are

==== ===== ==== ==== ==== ==== ==== ==== ====
D2h   A1g  B1g  B2g  B3g  A1u  B1u  B2u  B3u
==== ===== ==== ==== ==== ==== ==== ==== ====
A1g   A1g
B1g   B1g  A1g
B2g   B2g  B3g  A1g
B3g   B3g  B2g  B1g  A1g
A1u   A1u  B1u  B2u  B3u  A1g
B1u   B1u  A1u  B3u  B2u  B1g  A1g
B2u   B2u  B3u  A1u  B1u  B2g  B3g  A1g
B3u   B3u  B2u  B1u  A1u  B3g  B2g  B1g  A1g
==== ===== ==== ==== ==== ==== ==== ==== ====

The multiplication table for XOR operator is

==== ===== ==== ==== ==== ==== ==== ==== ====
XOR   000  001  010  011  100  101  110  111
==== ===== ==== ==== ==== ==== ==== ==== ====
000   000
001   001  000
010   010  011  000
011   011  010  001  000
100   100  101  110  111  000
101   101  100  111  110  001  000
110   110  111  100  101  010  011  000
111   111  110  101  100  011  010  001  000
==== ===== ==== ==== ==== ==== ==== ==== ====

Comparing the two table, we notice that the two tables can be changed to
each other with the mapping

==== ===== =====
D2h   XOR   ID
==== ===== =====
A1g   000    0
B1g   001    1
B2g   010    2
B3g   011    3
A1u   100    4
B1u   101    5
B2u   110    6
B3u   111    7
==== ===== =====

The XOR operator and the D2h subgroups have the similar relationships.
We therefore use the XOR operator ID to assign the irreps (see
:file:`pyscf/symm/param.py`).

==== ===== =====  ==== ===== =====  ==== ===== =====
C2h   XOR   ID    C2v   XOR   ID    D2    XOR   ID
==== ===== =====  ==== ===== =====  ==== ===== =====
Ag    00    0     A1    00    0     A1    00    0
Bg    01    1     A2    01    1     B1    01    1
Bg    10    2     B1    10    2     B2    10    2
Bg    11    3     B2    11    3     B3    11    3
==== ===== =====  ==== ===== =====  ==== ===== =====

==== ===== =====  ==== ===== =====  ==== ===== =====
Cs    XOR   ID    Ci    XOR   ID    C2    XOR   ID
==== ===== =====  ==== ===== =====  ==== ===== =====
A\'    0    0     Ag     0    0     A      0    0
B\"    1    1     Au     1    1     B      1    1
==== ===== =====  ==== ===== =====  ==== ===== =====

To easily get the relationship between the linear molecule symmetry
and D2h/C2v, the ID for irreducible representations of linear molecule
symmetry are chosen as (see :file:`pyscf/symm/basis.py`)

==================== ===  ============== ===
:math:`D_{\infty h}` ID   :math:`D_{2h}` ID
-------------------- ---  -------------- ---
A1g                   0    Ag            0      
A2g                   1    B1g           1      
A1u                   5    B1u           5      
A2u                   4    Au            4      
E1gx                  2    B2g           2      
E1gy                  3    B3g           3      
E1uy                  6    B2u           6      
E1ux                  7    B3u           7      
E2gx                  10   Ag            0      
E2gy                  11   B1g           1      
E2ux                  15   B1u           5      
E2uy                  14   Au            4      
E3gx                  12   B2g           2     
E3gy                  13   B3g           3     
E3uy                  16   B2u           6     
E3ux                  17   B3u           7     
E4gx                  20   Ag            0     
E4gy                  21   B1g           1     
E4ux                  25   B1u           5     
E4uy                  24   Au            4     
E5gx                  22   B2g           2     
E5gy                  23   B3g           3     
E5uy                  26   B2u           6     
E5ux                  27   B3u           7     
==================== ===  ============== ===

and

==================== === ============== ===
:math:`C_{\infty v}` ID  :math:`C_{2v}` ID
-------------------- --- -------------- ---
A1                   0   A1              0
A2                   1   A2              1
E1x                  2   B1              2
E1y                  3   B2              3
E2x                  10  A1              0
E2y                  11  A2              1
E3x                  12  B1              2
E3y                  13  B2              3
E4x                  20  A1              0
E4y                  21  A2              1
E5x                  22  B1              2
E5y                  23  B2              3
==================== === ============== ===

So that, the subduction from linear molecule symmetry to D2h/C2v can be
achieved by the modular operation ``%10``.

In many output messages, the irreducible representations are labeld with
the IDs instead of the irreps' symbols.  We can use
:func:`symm.addons.irrep_id2name` to convert the ID to irreps' symbol,
e.g.::

  >>> from pyscf import symm
  >>> [symm.irrep_id2name('Dooh', x) for x in [7, 6, 0, 10, 11, 0, 5, 3, 2, 5, 15, 14]]
  ['E1ux', 'E1uy', 'A1g', 'E2gx', 'E2gy', 'A1g', 'A1u', 'E1gy', 'E1gx', 'A1u', 'E2ux', 'E2uy']


SCF
---
To control the HF determinant symmetry,  one can assign occupancy for
particular irreps, e.g.

.. literalinclude:: ../../examples/scf/30-hf_with_irrep_nelec.py

FCI
---
FCI wavefunction symmetry can be controlled by initial guess.  Function
:func:`fci.addons.symm_initguess` can be used to generate the FCI
initial guess with the right symmetry.

CASSCF
------

CCSD
----
Symmetry are not supported in CCSD, MP2.


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


Modify Hamiltonian
==================

PySCF supports user-defined Hamiltonian for many modules.  To define
Hamiltonian for Hartree-Fock, CASSCF, MP2, CCSD, etc, one just need
replace the attributes :attr:`get_hcore` :attr:`get_ovlp`  and
:attr:`_eri` of SCF class for new Hamiltonian.  E.g. the user-defined
Hamiltonian for Hartree-Fock

.. literalinclude:: ../../examples/scf/40-hf_with_given_hamiltonian.py

and the user-defined Hamiltonian for CASSCF

.. literalinclude:: ../../examples/mcscf/40-casscf_with_given_hamiltonian.py


CASSCF solver
=============

Initial guess
-------------

Symmetry broken
---------------

DMRG solver
-----------


Callback
========


tools
=====


Others
======

Be careful with the warning messages on the screen!

