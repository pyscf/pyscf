.. _symm:

symm -- Point group symmetry and spin symmetry
**********************************************

.. automodule:: pyscf.symm

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
Au    10    2     B1    10    2     B2    10    2
Bu    11    3     B2    11    3     B3    11    3
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


Enabling symmetry in other module
---------------------------------

* SCF

  To control the HF determinant symmetry,  one can assign occupancy for
  particular irreps, e.g.

.. literalinclude:: ../../examples/scf/13-symmetry.py

* FCI

  FCI wavefunction symmetry can be controlled by initial guess.  Function
  :func:`fci.addons.symm_initguess` can generate the FCI initial guess with the
  right symmetry.

* MCSCF

  The symmetry of active space in the CASCI/CASSCF calculations can controlled
  
.. literalinclude:: ../../examples/mcscf/21-active_space_symmetry.py 

* MP2 and CCSD

  Point group symmetry are not supported in CCSD, MP2.


Program reference
=================

geom
----


.. automodule:: pyscf.symm.geom
   :members:

basis
-----

.. automodule:: pyscf.symm.basis
   :members:


addons
------

.. automodule:: pyscf.symm.addons
   :members:
 

Clebsch Gordon coefficients
---------------------------

.. automodule:: pyscf.symm.cg
   :members:


