#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pathlib import Path
from pyscf import gto
from pyscf import scf, dft

'''
As a Python script, you can use any Python trick to create your calculation.
In this example, we read the molecule geometry from another file.
'''

mol = gto.Mole()
mol.verbose = 5
mol.atom = (Path(__file__).resolve().with_name('glycine.xyz')).read_text()
mol.basis = '6-31g*'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

mf = dft.RKS(mol)
mf.kernel()
