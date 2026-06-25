"""
Take ASE Diamond structure, input into PySCF and run
"""

import sys
import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

verify_windows = '--pyscf-verify-windows' in sys.argv
try:
    from pyscf.pbc.tools import pyscf_ase
    import ase
    import ase.lattice
    from ase.lattice.cubic import Diamond
except (ModuleNotFoundError, RuntimeError):
    if verify_windows:
        # This example requires the optional ASE interface.
        print('Skipping PBC init-from-ASE example during Windows verification because ASE is not installed.')
        raise SystemExit(0)
    raise


ase_atom=Diamond(symbol='C', latticeconstant=3.5668)
print(ase_atom.get_volume())

cell = pbcgto.Cell()
cell.verbose = 5
cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a=ase_atom.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build()

mf=pbcdft.RKS(cell)

mf.xc='lda,vwn'

print(mf.kernel()) # [10,10,10]: -44.8811199336

