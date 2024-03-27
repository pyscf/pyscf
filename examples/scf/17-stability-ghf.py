#!/usr/bin/env python
#
# Author: Shirong Wang <srwang20@fudan.edu.cn>
#

'''
SCF wavefunction stability analysis

Comparison between Davidson and exact (slow) approach, for all kinds of stability,
especially real GHF internal, GHF real2complex, and complex GHF internal.
'''

from pyscf import gto, scf
from pyscf.scf import stability_slow, stability
from pyscf.scf.stability import stable_opt_internal
from pyscf.soscf import newton_ah
import numpy

# Simple trivial example: stable in each check

mol = gto.M(atom='''H 0.0 0.0 0.0; H 0.0 0.0 0.7''', basis='cc-pvdz', verbose=4).build()
mf = mol.RHF().run()
stability.rhf_internal(mf)
stability.rhf_external(mf)
stability_slow.rhf_internal(mf)
stability_slow.rhf_external(mf)

mf2 = mf.to_uhf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.uhf_internal(mf2)
#stability.uhf_internal(mf2, with_symmetry=False)
stability.uhf_external(mf2)
stability_slow.uhf_internal(mf2)
stability_slow.uhf_external(mf2)

mf2 = mf.to_ghf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.ghf_real(mf2, with_symmetry=True, tol=1e-6)
stability.ghf_real(mf2, with_symmetry=False, tol=1e-6)
stability_slow.ghf_real(mf2)

mf2.kernel(dm0 = mf2.make_rdm1()+0.0j)
stability.ghf_complex(mf2)
stability_slow.ghf_complex(mf2)


# H3, GHF real2complex stable
# from JCP 142, 154109 (2015)

PI = numpy.pi
N = 3
L = 1.0
R = L / (2*numpy.sin(PI / N ))
theta = [ (-0.5*PI + i*2*PI/N) for i in range(N)]
mol = gto.M(
atom = [['H', [R*numpy.cos(t), R*numpy.sin(t), 0.0]] for t in theta],
basis = 'cc-pvdz',
spin = 1, verbose = 4).build()
mf = scf.UHF(mol)
dm1 = numpy.zeros((15,15)), numpy.zeros((15,15))
dm1[0][1,1] = 1.0
dm1[0][6,6] = 1.0
dm1[1][10,10] = 0.3
mf.kernel(dm0=dm1)
mf.analyze()
_, mo = mf.stability(external=True)

mf2 = mf.to_ghf()
dm2 = mf2.make_rdm1(mo_coeff=mo)
mf2.kernel(dm0 = dm2)
mf2.analyze()
#stability.ghf_real(mf2, with_symmetry=True, nroots=6, tol=1e-6)
stability.ghf_real_internal(mf2, with_symmetry=False, nroots=6, tol=1e-6)
stability.ghf_real2complex(mf2, with_symmetry=False, nroots=6, tol=1e-6)

# tetrahedral H4
# from JCTC 2018, 14, 2, 649–659

L = 1.5
R = L / (2*numpy.sin(PI / N ))
theta = [ (-0.5*PI + i*2*PI/N) for i in range(N)]
mol = gto.M(
atom = [['H', [R*numpy.cos(t), R*numpy.sin(t), 0.0]] for t in theta] + [['H', [0.0, 0.0, numpy.sqrt(L*L-R*R)]]],
basis = 'cc-pvdz',
spin = 0, verbose = 4).build()
mf = scf.UHF(mol)
dm1 = numpy.zeros((20,20)), numpy.zeros((20,20))
dm1[0][1,1] = 1.0
dm1[0][6,6] = 1.0
dm1[1][11,11] = 1.0
dm1[1][16,16] = 1.0
mf.kernel(dm0=dm1)
mf.analyze()
_, mo = mf.stability(external=True)
dma, dmb = mf.make_rdm1()
m = dma - dmb
s = mf.get_ovlp()
Tzz = numpy.trace(numpy.einsum('ij,jk,kl,lm->im',m,s,m,s))
print('Tzz', Tzz)

def get_T_ghf(dm, s):
    nao = dm.shape[0]//2
    dmaa = dm[:nao,:nao]
    dmab = dm[:nao,nao:]
    dmba = dm[nao:,:nao]
    dmbb = dm[nao:,nao:]
    Txx = numpy.trace(numpy.einsum('ij,jk,kl,lm->im',dmab+dmba,s,dmab+dmba,s))
    Tyy = -numpy.trace(numpy.einsum('ij,jk,kl,lm->im',dmab-dmba,s,dmab-dmba,s))
    Tzz = numpy.trace(numpy.einsum('ij,jk,kl,lm->im',dmaa-dmbb,s,dmaa-dmbb,s))
    return Txx, Tyy, Tzz

mf2 = mf.to_ghf()
dm2 = mf2.make_rdm1(mo_coeff=mo)
mf2.kernel(dm0 = dm2)
mf2.analyze()
mf2.stability()
#stability.ghf_real(mf2, with_symmetry=True, nroots=6, tol=1e-6)
#stability.ghf_real_internal(mf2, with_symmetry=False, nroots=6, tol=1e-6)
#stability.ghf_real2complex(mf2, with_symmetry=False, nroots=6, tol=1e-6)
dm_rghf = mf2.make_rdm1()
Txx, Tyy, Tzz = get_T_ghf(dm_rghf, s)
print('Txx Tyy Tzz ', Txx, Tyy, Tzz)


mf2.kernel(dm0 = mf2.make_rdm1() + 0.0j)
#mo = mf2.stability()
mo = stability.ghf_complex(mf2, with_symmetry=False, tol=1e-6)
dm_cghf = mf2.make_rdm1()
Txx, Tyy, Tzz = get_T_ghf(dm_cghf, s)
print('Txx Tyy Tzz ', Txx, Tyy, Tzz)
stability_slow.ghf_complex(mf2)
exit()
dm2 = mf2.make_rdm1(mo_coeff=mo)
mf2.kernel(dm0 = dm2)
mf2.analyze()
mf2.stability()


