#!/usr/bin/env python
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import cc

# 1D anti-PBC Hubbard model at half filling

def ccsdsolver(fock, eri, nocc):
    mol = gto.M()
    fake_hf = scf.RHF(mol)
    fake_hf._eri = eri

    mycc = cc.ccsd.CC(fake_hf)

    # specify the problem size
    mycc.nmo = fock.shape[0]
    mycc.nocc = nocc

    # hack the integral transformation function to insert our hamiltonian
    def my_ao2mo(mo):
        mo = numpy.eye(mycc.nmo)
        eris = cc.ccsd._ERIS(mycc, mo)
        eris.fock = fock
        return eris
    mycc.ao2mo = my_ao2mo

    return mycc

n = 12
numpy.random.seed(1)
eri_on_mo = ao2mo.restore(8, numpy.random.random((n,n,n,n)), n)
fock_on_mo = numpy.random.random((n,n))
fock_on_mo = fock_on_mo + fock_on_mo.T
for i in range(n):
    fock_on_mo[i,i] += i*10

mycc = ccsdsolver(fock_on_mo, eri_on_mo, 2)
#NOTE: switch on DIIS early, otherwise the CCSD might have converge issue
mycc.diis_start_cycle = 0
mycc.diis_start_energy_diff = 1e2
mycc.verbose = 4
ecc = mycc.kernel()[0]
print('CCSD correlation energy = %.15g' % ecc)
