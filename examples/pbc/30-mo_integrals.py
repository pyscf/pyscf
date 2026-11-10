#!/usr/bin/env python

'''
MO integrals in PBC code
'''


import numpy
from pyscf import ao2mo
from pyscf.pbc import gto, scf, tools, df

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

#
# MO integrals at Gamma point.
#
mf = scf.RHF(cell).run()
eri = mf.with_df.ao2mo(mf.mo_coeff)
#
# These integrals are real. They can be transformed under the 8-fold
# permutation symmetry.
#
eri = ao2mo.restore(8, eri, mf.mo_coeff.shape[1])


#
# MO integrals at arbitrary k-point.
#

nk = [2,2,2]
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts).run()

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

# Compare AO-to-MO transformations from the three density-fitting backends
# against an explicit contraction at a momentum-conserving k-point quartet.
indices = (0, 1, 2, kconserv[0, 1, 2])
quartet = [kpts[i] for i in indices]
coeffs = [kmf.mo_coeff[i] for i in indices]
for fitting_class in (df.AFTDF, df.GDF, df.FFTDF):
    fitting = fitting_class(cell, kpts)
    eri_ao = fitting.get_eri(quartet).reshape([cell.nao] * 4)
    explicit = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao,
                            coeffs[0].conj(), coeffs[1],
                            coeffs[2].conj(), coeffs[3], optimize=True)
    transformed = fitting.ao2mo(coeffs, quartet).reshape([nmo] * 4)
    print(fitting_class.__name__, 'AO-to-MO difference:',
          abs(transformed - explicit).max())
