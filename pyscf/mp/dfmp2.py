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
from pyscf.mp import mp2
#from pyscf.mp.mp2 import make_rdm1, make_rdm2, make_rdm1_ao

def kernel(mp, eris=None, with_t2=False, verbose=None):

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    moidx = mp.get_frozen_mask()
    mo_coeff = mp.mo_coeff[:,moidx]
    mo_e = mp.mo_energy[moidx]
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]

    t2 = None
    emp2 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        for i in range(nocc):
            buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                            qov).reshape(nvir,nocc,nvir)
            gi = numpy.array(buf, copy=False)
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
            emp2 -= numpy.einsum('jab,jba', t2i, gi)

    return emp2, t2


class DFMP2(mp2.MP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if hasattr(mf, 'with_df') and mf.with_df:
            self.with_df = None
        else:
            self.with_df = df.DF(mol)
            self.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)

    @lib.with_doc(mp2.MP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=True):
        return mp2.MP2.kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        if self.with_df is None:
            with_df = self._scf.with_df
        else:
            with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
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
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254500453)

    mf = scf.density_fit(scf.RHF(mol), 'weigend')
    mf.kernel()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203738031827)

    pt.frozen = 2
    emp2, t2 = pt.kernel()
    print(emp2 - -0.14433975122418313)
