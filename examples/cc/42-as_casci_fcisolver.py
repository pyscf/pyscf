#!/usr/bin/env python

'''
Using the CCSD method as the active space solver to compute an approximate
CASCI energy.

A wrapper is required to adapt the CCSD solver to CASCI fcisolver interface.
Inside the wrapper function, the CCSD code is the same as the example
40-ccsd_with_given_hamiltonian.py
'''

import numpy
from pyscf import gto, scf, cc, ao2mo, mcscf

class AsFCISolver(object):
    def __init__(self):
        self.mycc = None

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        nelec = numpy.sum(nelec)
        fakemol.nelectron = nelec
        fake_hf = scf.RHF(fakemol)
        fake_hf._eri = ao2mo.restore(8, h2, norb)
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        fake_hf.kernel()
        self.mycc = cc.CCSD(fake_hf)
        self.eris = self.mycc.ao2mo()
        e_corr, t1, t2 = self.mycc.kernel(eris=self.eris)
        l1, l2 = self.mycc.solve_lambda(t1, t2, eris=self.eris)
        e = fake_hf.e_tot + e_corr
        return e+ecore, [t1,t2,l1,l2]

    def make_rdm1(self, fake_ci, norb, nelec):
        mo = self.mycc.mo_coeff
        t1, t2, l1, l2 = fake_ci
        dm1 = reduce(numpy.dot, (mo, self.mycc.make_rdm1(t1, t2, l1, l2), mo.T))
        return dm1

    def make_rdm12(self, fake_ci, norb, nelec):
        mo = self.mycc.mo_coeff
        nmo = mo.shape[1]
        t1, t2, l1, l2 = fake_ci
        dm2 = self.mycc.make_rdm2(t1, t2, l1, l2)
        dm2 = numpy.dot(mo, dm2.reshape(nmo,-1))
        dm2 = numpy.dot(dm2.reshape(-1,nmo), mo.T)
        dm2 = dm2.reshape([nmo]*4).transpose(2,3,0,1)
        dm2 = numpy.dot(mo, dm2.reshape(nmo,-1))
        dm2 = numpy.dot(dm2.reshape(-1,nmo), mo.T)
        dm2 = dm2.reshape([nmo]*4)
        return self.make_rdm1(fake_ci, norb, nelec), dm2

    def spin_square(self, fake_ci, norb, nelec):
        return 0, 1

mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
            basis = 'ccpvdz',
            verbose = 4)
mf = scf.RHF(mol).run()
norb = mf.mo_coeff.shape[1]
nelec = mol.nelectron
mc = mcscf.CASCI(mf, norb, nelec)
mc.fcisolver = AsFCISolver()
mc.kernel()

