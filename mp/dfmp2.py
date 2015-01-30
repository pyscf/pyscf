#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
import tempfile
from functools import reduce
import numpy
import h5py

import pyscf.ao2mo
from pyscf.ao2mo import _ao2mo
import pyscf.lib.logger as logger


'''
spin-adapted MP2
t2[i,j,b,a] = (ia|jb) / D_ij^ab
'''

def kernel(mp, mo_energy, mo_coeff, nocc, auxbasis='weigend', verbose=None):
    cderi = mp.ao2mo(mo_coeff, nocc)
    nvir = len(mo_energy) - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    t2 = None #numpy.empty((nocc,nocc,nvir,nvir))
    emp2 = 0
    for i in range(nocc):
        djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
        gi = numpy.einsum('pa,pjb->jba', cderi[:,i], cderi)
        t2i = (gi.ravel()/djba).reshape(nocc,nvir,nvir)
        #t2[i] = t2i
        # 2*ijab-ijba
        theta = gi*2 - gi.transpose(0,2,1)
        emp2 += numpy.einsum('jab,jab', t2i, theta)

    return emp2, t2


class MP2(object):
    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.auxbasis = 'weigend'

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
        logger.log(self, 'RMP2 energy = %.15g', self.emp2)
        return self.emp2, self.t2

    # Note return cderi_pov array[auxbas,nocc*nvir]
    def ao2mo(self, mo_coeff, nocc):
        import pyscf.df
        log = logger.Logger(self.stdout, self.verbose)
        time0 = (time.clock(), time.time())
        log.debug('transform (L|ia)')
        nmo = mo_coeff.shape[1]
        if hasattr(self._scf, '_cderi') and self._scf._cderi is not None:
            cderi = self._scf._cderi
        else:
            cderi = pyscf.df.incore.cholesky_eri(mol, auxbasis=self.auxbasis,
                                                 verbose=self.verbose)
        klshape = (0, nocc, nocc, nmo-nocc)
        cderimo = _ao2mo.nr_e2_(cderi, mo_coeff, klshape, aosym='s2kl', mosym='s1')
        time1 = log.timer('Integral transformation', *time0)
        return cderimo.reshape(-1,nocc,nmo-nocc)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.dfhf.RHF(mol)
    print(mf.scf())

    pt = MP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204019967288338)

