#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
density fitting MP2,  3-center integrals incore.
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
#from pyscf.mp.mp2 import make_rdm1, make_rdm2, make_rdm1_ao


# the MO integral for MP2 is (ov|ov). The most efficient integral
# transformation is
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)

def kernel(mp, mo_energy, mo_coeff, nocc, ioblk=256, verbose=None):
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc

    eia = lib.direct_sum('i-a->ia', mo_energy[:nocc], mo_energy[nocc:])
    t2 = None
    emp2 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        for i in range(nocc):
            buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                            qov).reshape(nvir,nocc,nvir)
            gi = numpy.array(buf, copy=False)
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,2,0)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            # 2*ijab-ijba
            theta = gi*2 - gi.transpose(0,2,1)
            emp2 += numpy.einsum('jab,jab', t2i, theta)

    return emp2, t2


class MP2(lib.StreamObject):
    def __init__(self, mf):
        self.mol = mf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self._scf = mf
        if hasattr(mf, 'with_df') and mf.with_df:
            self.with_df = None
        else:
            self.with_df = df.DF(mol)
            self.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)

        self.emp2 = None
        self.t2 = None

    def kernel(self, mo_energy=None, mo_coeff=None, nocc=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if nocc is None:
            nocc = self.mol.nelectron // 2

        self.emp2, self.t2 = \
                kernel(self, mo_energy, mo_coeff, nocc, verbose=self.verbose)
        logger.log(self, 'DF-RMP2 energy = %.15g', self.emp2)
        return self.emp2, self.t2

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        if self.with_df is None:
            with_df = self._scf.with_df
        else:
            with_df = self.with_df
        for eri1 in with_df.loop():
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

#    def make_rdm1(self, t2=None):
#        if t2 is None: t2 = self.t2
#        return make_rdm1(self, t2, self.verbose)
#
#    def make_rdm2(self, t2=None):
#        if t2 is None: t2 = self.t2
#        return make_rdm2(self, t2, self.verbose)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    pt = MP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254500453)

    mf = scf.density_fit(scf.RHF(mol), 'weigend')
    mf.kernel()
    pt = MP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203738031827)
