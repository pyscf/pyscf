#!/usr/bin/env python

'''
MO integrals in PBC code
'''


import numpy
from pyscf.pbc import gto, scf, tools

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 4,
)

nk = [2,2,2]
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts, exxdiv=None)
kmf.kernel()


nmo = kmf.mo_coeff[0].shape[1]
kconserv = tools.get_kconserv(cell, kpts)
nkpts = len(kpts)
for kp in range(nkpts):
    for kq in range(nkpts):
        for kr in range(nkpts):
            ks = kconserv[kp,kq,kr]

            eri_kpt = kmf.with_df.ao2mo([kmf.mo_coeff[i] for i in (kp,kq,kr,ks)],
                                        [kpts[i] for i in (kp,kq,kr,ks)])
            eri_kpt = eri_kpt.reshape([nmo]*4)

