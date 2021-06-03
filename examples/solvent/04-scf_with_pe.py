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

The potfile required by this example can be generated with the script
04-pe_potfile_from_pyframe.py
'''

import pyscf
from pyscf import solvent

mol = pyscf.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
mf = mol.UKS()
mf.xc = 'b3lyp'
mf = solvent.PE(mf, '4NP_in_water/4NP_in_water.pot')
mf.run()

