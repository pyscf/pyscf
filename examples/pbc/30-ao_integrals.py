#!/usr/bin/env python

'''
Access AO integrals in PBC code
'''


from pyscf.pbc import gto, df
import numpy

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
# 1-electron (periodic) integrals can be obtained with pbc_intor function.
# Note .intor function computes integrals without PBC
#
overlap = cell.pbc_intor('cint1e_ovlp_sph')
kinetic = cell.pbc_intor('cint1e_kin_sph')

#
# In PBC, 2e integrals are all computed with 3-ceter integrals, ie from
# density fitting module.  Without kpts argument, the DF gives gamma point AO
# 2e-integrals.  Permutation symmetry is not considered
#
mydf = df.FFTDF(cell)
eri = mydf.get_eri()
print('ERI shape (%d,%d)' % eri.shape)

kpts = numpy.array((
    (-.5, .5, .5),
    ( .5,-.5, .5),
    ( .5, .5,-.5),
    ( .5, .5, .5),
))
#
# Note certain translational symmetry is required:
#       kj-ki+kl-kk = 2n\pi
#
ki, kj, kk = kpts[:3]
kl = kk + ki - kj

mydf.kpts = kpts
eri = mydf.get_eri((ki,kj,kk,kl))
print('ERI shape (%d,%d)' % eri.shape)

#
# Use MDF method to get better accuracy for valence basis.
#
mydf = df.MDF(cell, kpts)
mydf.auxbasis = 'weigend'
eri = mydf.get_eri((ki,kj,kk,kl))
print('ERI shape (%d,%d)' % eri.shape)

#
# Using .loop method to access the 3-index ERI tensor at gamma point.  In the
# 3-index tensor Lpq, orbital basis index (p,q) are compressed (p>=q)
#
mydf = df.FFTDF(cell)
eri_3d = numpy.vstack([Lpq.copy() for Lpq in mydf.loop()])
# Test 2-e integrals
eri = numpy.dot(eri_3d.T, eri_3d)
print(abs(eri - mydf.get_eri(compact=True)).max())

mydf = df.MDF(cell)
eri_3d = numpy.vstack([Lpq.copy() for Lpq in mydf.loop()])
# Test 2-e integrals
eri = numpy.dot(eri_3d.T, eri_3d)
print(abs(eri - mydf.get_eri(compact=True)).max())

#
# Using .sr_loop method to access the 3-index tensor of gaussian density
# fitting (GDF) for arbitrary k-points
# The same can be achieved by using range-separated density fitting (RSDF).
# To do so, simply change 'df.DF' to be 'df.RSDF' below.
# see '35-range_separated_density_fit.py' for more details of RSDF.
#
nao = cell.nao_nr()
mydf = df.DF(cell, kpts=kpts)
eri_3d_kpts = []
for i, kpti in enumerate(kpts):
    eri_3d_kpts.append([])
    for j, kptj in enumerate(kpts):
        eri_3d = []
        for LpqR, LpqI, sign in mydf.sr_loop([kpti,kptj], compact=False):
            eri_3d.append(LpqR+LpqI*1j)
        eri_3d = numpy.vstack(eri_3d).reshape(-1,nao,nao)
        eri_3d_kpts[i].append(eri_3d)
# Test 2-e integrals
# Note, Coulomb integrals are positive definition in this example. The code
# below does not work for 2D pbc system.
eri = numpy.einsum('Lpq,Lrs->pqrs', eri_3d_kpts[0][3], eri_3d_kpts[3][0])
print(abs(eri - mydf.get_eri([kpts[0],kpts[3],kpts[3],kpts[0]]).reshape([nao]*4)).max())

#
# Using .get_ao_pairs_G method to access the 3-index tensor of FFTDF method
# for arbitrary k-points. Be careful with the k-point wrap-around treatment.
#
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc import tools
nao = cell.nao_nr()
mydf = df.FFTDF(cell, kpts=kpts)
Lpq_kpts = []
for i, kpti in enumerate(kpts):
    Lpq_kpts.append([])
    for j, kptj in enumerate(kpts):
        q = kptj - kpti
        coulG = tools.get_coulG(cell, q)
        ngrids = len(coulG)
        ao_pairs_G = mydf.get_ao_pairs_G([kpti,kptj], q, compact=False)
        ao_pairs_G *= numpy.sqrt(coulG*cell.vol/ngrids**2).reshape(-1,1)
        Lpq_kpts[i].append(ao_pairs_G.reshape(-1,nao,nao))

kconserv = get_kconserv(cell, kpts)
Lrs_kpts = []
for i, kpti in enumerate(kpts):
    Lrs_kpts.append([])
    for j, kptj in enumerate(kpts):
        Lrs_kpts[i].append([])
        q = kptj - kpti
        coulG = tools.get_coulG(cell, q)
        ngrids = len(coulG)
        for k, kptk in enumerate(kpts):
            # Handle the wrap-around k-points
            l = kconserv[i,j,k]
            kptl = kpts[l]
            ao_pairs_invG = mydf.get_ao_pairs_G([-kptk,-kptl], q, compact=False).conj()
            ao_pairs_invG *= numpy.sqrt(coulG*cell.vol/ngrids**2).reshape(-1,1)
            Lrs_kpts[i][j].append(ao_pairs_invG.reshape(-1,nao,nao))
# Test 2-e integrals
eri = numpy.einsum('Lpq,Lrs->pqrs', Lpq_kpts[0][3], Lrs_kpts[0][3][3])
print(abs(eri - mydf.get_eri([kpts[0],kpts[3],kpts[3],kpts[0]]).reshape([nao]*4)).max())
