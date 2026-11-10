#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example of using polarizable embedding model in the mean-field
calculations. This example requires the cppe library

GitHub:      https://github.com/maxscheurer/cppe
Code:        10.5281/zenodo.3345696
Publication: https://doi.org/10.1021/acs.jctc.9b00758

The CPPE library can be installed via:
pip install git+https://github.com/maxscheurer/cppe.git

The first calculation creates a small potential file for one water molecule.
An optional external potential can be generated with 04-pe_potfile_from_pyframe.py.
'''

from pathlib import Path
import pyscf
from pyscf import lib, solvent

mol = pyscf.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
# A potential contains coordinates, permanent multipoles, polarizabilities,
# and exclusion lists for interactions within the embedding water molecule.
# Keep the temporary file alive throughout the embedded SCF calculation.
with lib.NamedTemporaryFile() as potfile:
    potfile.write(b'''!
@COORDINATES
3
AA
O     3.53300000    2.99600000    0.88700000      1
H     4.11100000    3.13200000    1.63800000      2
H     4.10500000    2.64200000    0.20600000      3
@MULTIPOLES
ORDER 0
3
1     -0.67444000
2      0.33722000
3      0.33722000
@POLARIZABILITIES
ORDER 1 1
3
1      5.73935000     0.00000000     0.00000000     5.73935000     0.00000000     5.73935000
2      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
3      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
EXCLISTS
3 3
1   2  3
2   1  3
3   1  2
''')
    potfile.flush()
    mf = solvent.PE(mol.RHF(), {"potfile": potfile.name}).run()
    print('PE-RHF total energy', mf.e_tot)
    print('Embedding contribution', mf.with_solvent.e)

# If the external data from 04-pe_potfile_from_pyframe.py are available,
# the same interface can read that potential directly for a DFT calculation.
external_potential = Path('4NP_in_water/4NP_in_water.pot')
if external_potential.is_file():
    mf = mol.UKS()
    mf.xc = 'b3lyp'
    mf = solvent.PE(mf, str(external_potential)).run()
