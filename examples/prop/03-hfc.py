#!/usr/bin/env python

'''
Computing hyperfine coupling tensor
'''

from pyscf import gto, scf, dft
from pyscf.prop import hfc
mol = gto.M(atom='''
            C 0 0 0
            N 0 0 1.1747
            ''',
            basis='ccpvdz', spin=1, charge=0, verbose=3)
mf = scf.UHF(mol).run()
gobj = hfc.uhf.HFC(mf).set(verbose=4)

# 2-electron SOC for para-magnetic term. See also the example 02-g_tensor.py
gobj.para_soc2e = 'SSO+SOO'

# Disable Koseki effective SOC charge
gobj.so_eff_charge = False

gobj.kernel()

