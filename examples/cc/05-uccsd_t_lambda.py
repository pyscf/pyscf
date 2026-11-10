#!/usr/bin/env python

"""Compare CCSD(T) triples corrections and lambda updates for H2O.

First compare triples corrections, including a randomized spin-representation
check. Then compare restricted and unrestricted lambda updates.

Apply two restricted lambda updates, then reproduce the second update using
unrestricted spin blocks. The printed differences should be close to zero.
This illustrates the low-level update API; see 01-lambda.py for solving the
lambda equations to convergence with kernel().
"""

from pyscf import gto
from pyscf import scf
from pyscf import cc
import numpy
from pyscf.cc import uccsd_t, gccsd_t_slow, uccsd_t_lambda

# Triples energy corrections
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.757 , .587)],
    [1 , (0. ,  .757 , .587)]]

mol.basis = '631g'
mol.build()
rhf = scf.RHF(mol)
rhf.conv_tol = 1e-14
rhf.scf()
mcc = cc.CCSD(rhf)
mcc.conv_tol = 1e-12
mcc.ccsd()
t1a = t1b = mcc.t1
t2ab = mcc.t2
t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
mycc = cc.UCCSD(scf.addons.convert_to_uhf(rhf))
eris = mycc.ao2mo()
e3a = uccsd_t.kernel(mycc, eris, (t1a,t1b), (t2aa,t2ab,t2bb))
print('Closed-shell triples correction difference:', e3a - -0.00099642337843278096)

# Numerical consistency check with deliberately randomized orbitals and
# amplitudes. This is not a physical energy calculation: it compares the
# unrestricted and generalized spin representations of the triples correction.
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.757 , .587)],
    [1 , (0. ,  .757 , .587)]]
mol.spin = 2
mol.basis = '3-21g'
mol.build()
mf = scf.UHF(mol).run(conv_tol=1e-14)
nao, nmo = mf.mo_coeff[0].shape
numpy.random.seed(10)
mf.mo_coeff = numpy.random.random((2,nao,nmo))

numpy.random.seed(12)
nocca, noccb = mol.nelec
nmo = mf.mo_occ[0].size
nvira = nmo - nocca
nvirb = nmo - noccb
t1a  = .1 * numpy.random.random((nocca,nvira))
t1b  = .1 * numpy.random.random((noccb,nvirb))
t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
t2aa = t2aa - t2aa.transpose(0,1,3,2)
t2aa = t2aa - t2aa.transpose(1,0,2,3)
t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
t2bb = t2bb - t2bb.transpose(0,1,3,2)
t2bb = t2bb - t2bb.transpose(1,0,2,3)
t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
t1 = t1a, t1b
t2 = t2aa, t2ab, t2bb
mycc = cc.UCCSD(mf)
eris = mycc.ao2mo(mf.mo_coeff)
e3a = uccsd_t.kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
print('Random-amplitude UCCSD(T) reference difference:', e3a - 9877.2780859693339)

mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
eris = mycc.ao2mo()
t1 = mycc.spatial2spin(t1, eris.orbspin)
t2 = mycc.spatial2spin(t2, eris.orbspin)
et = gccsd_t_slow.kernel(mycc, eris, t1, t2)
print('Random-amplitude GCCSD(T) reference difference:', et - 9877.2780859693339)
print('Unrestricted/generalized triples difference:', e3a - et)

# Lambda updates on a separate closed-shell reference

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '631g'
mol.spin = 0
mol.build()
# A deliberately loose SCF reference also exercises noncanonical terms.
# Both representations below use exactly the same orbitals.
mf0 = mf = scf.RHF(mol).run(conv_tol=1)
mf = scf.addons.convert_to_uhf(mf)
mycc = cc.UCCSD(mf)
eris = mycc.ao2mo()

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
mycc0 = cc.CCSD(mf0)
eris0 = mycc0.ao2mo()
mycc0.kernel(eris=eris0)
t1 = mycc0.t1
t2 = mycc0.t2
imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
l1ref, l2ref = ccsd_t_lambda.update_lambda(mycc0, t1, t2, l1, l2, eris0, imds)

# Map restricted amplitudes into alpha/beta spin blocks. Same-spin
# doubles are obtained by antisymmetrizing the restricted doubles.
t1 = (t1, t1)
t2aa = t2 - t2.transpose(1,0,2,3)
t2 = (t2aa, t2, t2aa)
l1 = (l1, l1)
l2aa = l2 - l2.transpose(1,0,2,3)
l2 = (l2aa, l2, l2aa)
imds = uccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
l1, l2 = uccsd_t_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
print('Alpha-alpha antisymmetry difference:',
      abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
print('Beta-beta antisymmetry difference:',
      abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[2]).max())
print('Restricted/unrestricted singles difference:', abs(l1[0]-l1ref).max())
print('Restricted/unrestricted mixed-spin doubles difference:', abs(l2[1]-l2ref).max())
