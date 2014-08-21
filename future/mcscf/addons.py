#!/usr/bin/env python

import numpy
from pyscf import ao2mo
from pyscf.future.fci import direct_spin0 as fci_direct

def _make_rdm1_on_mo(casdm1, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    return dm1

# on AO representation
def make_rdm1(casscf, fcivec, mo):
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    casdm1 = fci_direct.make_rdm1(fcivec, ncas, nelecas)
    rdm1 = _make_rdm1_on_mo(casdm1, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))
    return rdm1

def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2

# on AO representation
def make_rdm12(casscf, fcivec, mo):
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    casdm1, casdm2 = fci_direct.make_rdm12(fcivec, ncas, nelecas)
    rdm1, rdm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))
    rdm2 = numpy.dot(mo, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo.T)
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
    rdm2 = numpy.dot(mo, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo.T)
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

if __name__ == '__main__':
    import scf
    import gto
    import mc1step
    import tools.ring

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['H', c] for c in tools.ring.make(6, 1.2)]
    mol.basis = {'H': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(mol, m, 6, 6)
    mc.verbose = 4
    emc, e_ci, fcivec, mo = mc.mc1step()
    emc += mol.nuclear_repulsion()
    print ehf, emc, emc-ehf
    print emc - -3.272089958

    rdm1 = make_rdm1(mc, fcivec, mo)
    rdm1, rdm2 = make_rdm12(mc, fcivec, mo)
    print rdm1
