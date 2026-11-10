#!/usr/bin/env python

"""Compare GCCSD(T) density matrices with restricted/unrestricted forms.

Use closed-shell H2O and open-shell H2O(2+) to compare spin blocks, then
reconstruct a GCCSD(T) total energy from its one- and two-particle RDMs.
The deliberately loose SCF references and single lambda updates in the
representation comparisons follow the original numerical demonstration;
those sections do not solve the triples-corrected lambda equations to convergence.
"""

from functools import reduce
import numpy
from pyscf.cc import gccsd_t_rdm
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import cc

# Closed-shell: compare spin-summed generalized and restricted RDMs.
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.957 , .587)],
    [1 , (0.2,  .757 , .487)]]
mol.basis = '631g'
mol.build()
mf0 = mf = scf.RHF(mol).run(conv_tol=1.)
mf = mf.to_ghf()

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
mycc0 = cc.CCSD(mf0)
eris0 = mycc0.ao2mo()
mycc0.kernel(eris=eris0)
t1 = mycc0.t1
t2 = mycc0.t2
imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
dm1ref = ccsd_t_rdm.make_rdm1(mycc0, t1, t2, l1, l2, eris0)
dm2ref = ccsd_t_rdm.make_rdm2(mycc0, t1, t2, l1, l2, eris0)

mycc = cc.GCCSD(mf)
eris = mycc.ao2mo()
t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
gdm1 = gccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris)
gdm2 = gccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris)
idxa = numpy.where(mycc.mo_coeff.orbspin == 0)[0]
idxb = numpy.where(mycc.mo_coeff.orbspin == 1)[0]

trdm1 = gdm1[idxa[:,None],idxa]
trdm1+= gdm1[idxb[:,None],idxb]
trdm2 = gdm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa]
trdm2+= gdm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb]
dm2ab = gdm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb]
trdm2+= dm2ab
trdm2+= dm2ab.transpose(2,3,0,1)
print('Closed-shell 1-RDM difference:',
      abs(trdm1 - dm1ref).max())
print('Closed-shell 2-RDM difference:',
      abs(trdm2 - dm2ref).max())

# Open-shell: compare individual generalized and unrestricted spin blocks.
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '631g'
mol.spin = 2
mol.charge = 2
mol.build()
mf0 = mf = scf.UHF(mol).run(conv_tol=1)
mf = scf.addons.convert_to_ghf(mf)

from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
mycc0 = cc.UCCSD(mf0)
eris0 = mycc0.ao2mo()
mycc0.kernel(eris=eris0)
t1 = mycc0.t1
t2 = mycc0.t2
imds = uccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
l1, l2 = uccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
dm1ref = uccsd_t_rdm.make_rdm1(mycc0, t1, t2, l1, l2, eris0)
dm2ref = uccsd_t_rdm.make_rdm2(mycc0, t1, t2, l1, l2, eris0)

mycc = cc.GCCSD(mf)
eris = mycc.ao2mo()
t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
gdm1 = gccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris)
gdm2 = gccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris)
idxa = numpy.where(mycc.mo_coeff.orbspin == 0)[0]
idxb = numpy.where(mycc.mo_coeff.orbspin == 1)[0]

print('Open-shell alpha 1-RDM difference:',
      abs(dm1ref[0] - gdm1[idxa[:,None],idxa]).max())
print('Open-shell beta 1-RDM difference:',
      abs(dm1ref[1] - gdm1[idxb[:,None],idxb]).max())
print('Open-shell alpha-alpha 2-RDM difference:',
      abs(dm2ref[0] - gdm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa]).max())
print('Open-shell alpha-beta 2-RDM difference:',
      abs(dm2ref[1] - gdm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb]).max())
print('Open-shell beta-beta 2-RDM difference:',
      abs(dm2ref[2] - gdm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb]).max())

# Reconstruct the GCCSD(T) total energy from the density matrices.
ecc, t1, t2 = mycc.kernel(eris=eris)
e3ref = mycc.e_tot + mycc.ccsd_t()
l1, l2 = mycc.solve_lambda(eris=eris)
dm1 = gccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)
dm2 = gccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris=eris)
nao = mol.nao_nr()
mo_a = mf.mo_coeff[:nao]
mo_b = mf.mo_coeff[nao:]
nmo = mo_a.shape[1]
eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
orbspin = mf.mo_coeff.orbspin
sym_forbid = (orbspin[:,None] != orbspin)
eri[sym_forbid,:,:] = 0
eri[:,:,sym_forbid] = 0
hcore = scf.RHF(mol).get_hcore()
h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
e3 = numpy.einsum('ij,ji', h1, dm1)
e3+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
e3+= mol.energy_nuc()
print('RDM energy minus GCCSD(T) total energy:', e3 - e3ref)
