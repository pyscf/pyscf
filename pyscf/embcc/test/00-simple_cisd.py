#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CISD calculation.
'''
import numpy as np

import pyscf
import pyscf.scf
import pyscf.ci
import pyscf.cc
import pyscf.fci
import pyscf.ao2mo

#mol = pyscf.M(
#    atom = 'H 0 0 0; H 0 0 1.2',
#    basis = 'ccpvdz',
#    verbose=4)

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz',
    verbose=4
    )



mf = pyscf.scf.RHF(mol)
mf.run()

def get_ci_energy(cc, C0, C1, C2):

    # Intermediate normalization <CI|HF> = 1
    renorm = 1/C0
    C1 *= renorm
    C2 *= renorm

    a = cc.get_frozen_mask()
    o = cc.mo_occ[a] > 0
    v = cc.mo_occ[a] == 0
    #S = mf.get_ovlp()
    #C = cc.mo_coeff[:,a]

    eris = cc.ao2mo()
    F = eris.fock[o][:,v]
    e1 = 2*np.sum(F * C1)
    # Brillouin's theorem
    assert np.isclose(e1, 0)

    e2 = 2*np.einsum('ijab,iabj', C2, eris.ovvo, optimize=True)
    e2 -=  np.einsum('ijab,jabi', C2, eris.ovvo, optimize=True)

    e_ccsd = e1+e2
    return e_ccsd


#for i, e in enumerate(mf.mo_energy):
#    print(i, e)

active = [4,5,6,7,8,9]
frozen = [i for i in range(len(mf.mo_energy)) if i not in active]

cisd = pyscf.ci.CISD(mf, frozen=frozen)
cisd.run()
print('RCISD correlation energy: %.8e' % cisd.e_corr)

from pyscf import mcscf
cas = mcscf.CASCI(mf, 6, 6)
cas.canonicalization = False
Etot, Ecas, wf, mo_coeff, mo_energy = cas.kernel()

assert np.allclose(mf.mo_energy, cas.mo_energy)
assert np.allclose(mf.mo_coeff, cas.mo_coeff)

norb = 6
nelec = 6
cisdvec = pyscf.ci.cisd.from_fcivec(wf, norb, nelec)
C0, C1 ,C2 = cisd.cisdvec_to_amplitudes(cisdvec)
e_corr = get_ci_energy(cisd, C0, C1, C2)
print("CASCI: %e" % e_corr)

print("CASCI correlation energy: %.8e" % (Etot-mf.e_tot))


C0, C1 ,C2 = cisd.cisdvec_to_amplitudes(cisd.ci)
e_corr = get_ci_energy(cisd, C0, C1, C2)
print("CISD: %e" % e_corr)


# USE  FCI module instead...
fci = pyscf.fci.direct_spin0.FCISolver(mol)
C = mf.mo_coeff[:,active]
#C = cas.mo_coeff[:,active]
h1e = np.linalg.multi_dot((C.T, mf.get_hcore(), C))
#eri = pyscf.ao2mo.full(mol, C)
eri = pyscf.ao2mo.kernel(mol, C)
print(h1e.shape)
print(eri.shape)

h1eff, Ecore = cas.h1e_for_cas(mo_coeff=mf.mo_coeff, ncas=norb, ncore=4)

Efci, Wfci = fci.kernel(h1eff, eri, norb, nelec)
print(Efci)
print(Wfci.shape)

cisdvec = pyscf.ci.cisd.from_fcivec(Wfci, norb, nelec)
C0, C1 ,C2 = cisd.cisdvec_to_amplitudes(cisdvec)
e_corr = get_ci_energy(cisd, C0, C1, C2)
print("FCI: %e" % e_corr)



1/0

C0, C1 ,C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(ci, nmo, nocc)
e_corr = get_ci_energy(C0, C1, C2)
print(E_corr)

1/0


#cc = pyscf.cc.CCSD(mf)
#cc.run()
#
#eris = cc.ao2mo()
#eris_ovvo = eris.ovvo
#eris_ovvo2 = np.moveaxis(eris_ovvo, [0,1,2,3], [3,2,1,0])
#assert np.allclose(eris_ovvo, eris_ovvo2)
#
#T2 = cc.t2
#T2t = np.moveaxis(T2, [0,1,2,3], [1,0,3,2])
#assert np.allclose(T2, T2t)
#
#1/0

cc = pyscf.ci.CISD(mf)
cc.run()
print('RCISD correlation energy', cc.e_corr)

#C0, C1 ,C2 = cc.cisdvec_to_amplitudes(cc.ci)
C0, C1 ,C2 = pyscf.cisd.cisdvec_to_amplitudes(cc.ci)

e_cisd = get_ci_energy(C0, C1, C2)
print("%.10f, error=%.2e" % (e_cisd, e_cisd - cc.e_corr))

# --- FCI
import pyscf.fci

fci = pyscf.fci.FCI(mol, mf.mo_coeff)
e_fci, civec = fci.kernel()
e_fci_corr = e_fci - mf.e_tot

print(np.allclose(fci.mo_coeff, mf.mo_coeff))
print(np.allclose(fci.mo_occ, mf.mo_occ))
print("FCI: %f" % e_fci_corr)

norb = len(mf.mo_energy)
nelec = mol.nelectron
cisdvec = pyscf.ci.cisd.from_fcivec(civec, norb, nelec)
C0, C1 ,C2 = cc.cisdvec_to_amplitudes(cisdvec)

e_cisd = get_ci_energy(C0, C1, C2)
print("%.10f, error=%.2e" % (e_cisd, e_cisd - e_fci_corr))



