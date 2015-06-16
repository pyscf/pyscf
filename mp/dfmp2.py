#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
density fitting MP2,  3-center integrals incore.
'''

import time
import tempfile
import numpy
from pyscf.lib import logger
from pyscf import df


# the MO integral for MP2 is (ov|ov). The most efficient integral
# transformation is
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)

def kernel(mp, mo_energy, mo_coeff, nocc, ioblk=256, verbose=None):
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    auxmol = df.incore.format_aux_basis(mp.mol, mp.auxbasis)
    naoaux = auxmol.nao_nr()

    iolen = max(int(ioblk*1e6/8/(nvir*nocc)), 160)

    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    t2 = None
    emp2 = 0
    with mp.ao2mo(mo_coeff, nocc) as fov:
        for p0, p1 in prange(0, naoaux, iolen):
            logger.debug(mp, 'Load cderi block %d:%d', p0, p1)
            qov = numpy.array(fov[p0:p1], copy=False)
            for i in range(nocc):
                buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                                qov).reshape(nvir,nocc,nvir)
                djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
                gi = numpy.array(buf, copy=False)
                gi = gi.reshape(nvir,nocc,nvir).transpose(1,2,0)
                t2i = (gi.ravel()/djba).reshape(nocc,nvir,nvir)
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
        if hasattr(mf, 'auxbasis'):
            self.auxbasis = mf.auxbasis
        else:
            self.auxbasis = 'weigend'
        self._cderi = None
        self.ioblk = 256

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
                kernel(self, mo_energy, mo_coeff, nocc, self.ioblk,
                       verbose=self.verbose)
        logger.log(self, 'RMP2 energy = %.15g', self.emp2)
        return self.emp2, self.t2

    # MO integral transformation for cderi[auxstart:auxcount,:nao,:nao]
    def ao2mo(self, mo_coeff, nocc):
        time0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        cderi_file = tempfile.NamedTemporaryFile()
        df.outcore.general(self.mol, (mo_coeff[:,:nocc], mo_coeff[:,nocc:]),
                           cderi_file.name, auxbasis=self.auxbasis, verbose=log)
        time1 = log.timer('Integral transformation (P|ia)', *time0)
        return df.load(cderi_file)

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

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
    mf = scf.RHF(mol)
    mf.scf()
    pt = MP2(mf)
    pt.max_memory = .05
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254491987)

    mf = scf.density_fit(scf.RHF(mol))
    mf.scf()
    pt = MP2(mf)
    pt.max_memory = .05
    pt.ioblk = .05
    pt.verbose = 5
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)
