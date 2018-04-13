#!/usr/bin/env python
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import cc

def ccsdsolver(hcore, eri, nocc):
    mol = gto.M()
    fake_hf = scf.RHF(mol)
    fake_hf._eri = eri
    fake_hf.get_hcore = lambda *args: hcore

    nmo = hcore.shape[0]
    mo_coeff = numpy.eye(nmo)
    mo_occ = numpy.zeros(nmo)
    mo_occ[:nocc] = 2

    mycc = cc.ccsd.CCSD(fake_hf, mo_coeff=mo_coeff, mo_occ=mo_occ)
    #NOTE: switch on DIIS early, otherwise the CCSD might have converge issue
    mycc.diis_start_cycle = 0
    mycc.diis_start_energy_diff = 1e2
    return mycc

numpy.random.seed(1)
n = 12
eri_on_mo = ao2mo.restore(8, numpy.random.random((n,n,n,n)), n)
hcore_on_mo = numpy.random.random((n,n))
hcore_on_mo = hcore_on_mo + hcore_on_mo.T
hcore_on_mo += numpy.diag(numpy.arange(n)*10)

mycc = ccsdsolver(hcore_on_mo, eri_on_mo, 2)
mycc.verbose = 4
ecc = mycc.kernel()
print('CCSD correlation energy = %.15g' % mycc.e_corr)
