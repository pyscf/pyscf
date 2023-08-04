#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Symmetry is not immutable

In PySCF, symmetry is not built-in data structure.  Orbitals are stored in C1
symmetry.  The irreps and symmetry information are generated on the fly.
We can switch on symmetry for CASSCF solver even the Hartree-Fock is not
optimized with symmetry.
'''

mol = gto.Mole()
mol.build(
    atom = [['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
    basis = 'cc-pvdz',
)
mf = scf.RHF(mol)
mf.kernel()

mol.build(0, 0, symmetry = 'C2v')
mc = mcscf.CASSCF(mf, 6, 8)
mc.kernel()
