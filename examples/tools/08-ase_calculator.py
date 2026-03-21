#!/usr/bin/env python

'''
Using PySCF as an ASE Calculator

Any PySCF mean-field or post-mean-field method (e.g., DFT, HF, CCSD) can be
wrapped as an ASE-compatible calculator using the pyscf_ase interface.

More examples can be found in examples/pbc/09-ase_geometry_optimization.py and
examples/pbc/09-talk_to_ase.py
'''

import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase import Atoms

# Define molecules using ase.Atoms class, the ase_atoms_to_pyscf function can
# convert the geometry to pyscf input format
atoms = Atoms('N2', [(0, 0, 0), (0, 0, 1.2)])
mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis='ccpvdz')

# Example 1: Use a DFT method as an ASE calculator
mf = mol.RKS(xc='pbe0')
atoms.calc = PySCF(method=mf)

# Example 2: Use a post-HF method (CCSD) as an ASE calculator
atoms.calc = PySCF(method=mol.RHF().CCSD())
