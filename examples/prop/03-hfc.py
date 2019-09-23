#!/usr/bin/env python

from pyscf import gto, scf, dft
from pyscf.prop import hfc
mol = gto.M(atom='''
            C 0 0 0
            N 0 0 1.1747
            ''',
            basis='ccpvdz', spin=1, charge=0, verbose=3)
mf = scf.UHF(mol).run()
gobj = hfc.uhf.HFC(mf).set(verbose=4)
gobj.sso = True
gobj.soo = True
gobj.so_eff_charge = False
gobj.kernel()

