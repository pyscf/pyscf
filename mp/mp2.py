#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
import tempfile
from functools import reduce
import warnings
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo


'''
spin-adapted MP2
t2[i,j,a,b] = (ia|jb) / D_ij^ab
'''

# the MO integral for MP2 is (ov|ov). The most efficient integral
# transformation is
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = lib.direct_sum('i-a->ia', mo_energy[:nocc], mo_energy[nocc:])
    t2 = numpy.empty((nocc,nocc,nvir,nvir))
    emp2 = 0

    with mp.ao2mo(mo_coeff) as ovov:
        for i in range(nocc):
            gi = numpy.asarray(ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2[i] = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            # 2*ijab-ijba
            theta = gi*2 - gi.transpose(0,2,1)
            emp2 += numpy.einsum('jab,jab', t2[i], theta)

    return emp2, t2

# Need less memory
def make_rdm1_ao(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    dm1occ = numpy.zeros((nocc,nocc))
    dm1vir = numpy.zeros((nvir,nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    emp2 = 0
    with mp.ao2mo(mo_coeff) as ovov:
        for i in range(nocc):
            dajb = (eia[i].reshape(-1,1) +
                    eia.reshape(1,-1)).reshape(nvir,nocc,nvir)
            gi = numpy.asarray(ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = (gi/dajb.transpose(1,0,2)).reshape(nocc,nvir,nvir)
            # 2*ijab-ijba
            theta = gi*2 - gi.transpose(0,2,1)
            emp2 += numpy.einsum('jab,jab', t2i, theta)

            dm1vir += numpy.einsum('jca,jcb->ab', t2i, t2i) * 2 \
                    - numpy.einsum('jca,jbc->ab', t2i, t2i)
            dm1occ += numpy.einsum('iab,jab->ij', t2i, t2i) * 2 \
                    - numpy.einsum('iab,jba->ij', t2i, t2i)

    rdm1 = numpy.zeros((nmo,nmo))
# *2 for beta electron
    rdm1[:nocc,:nocc] =-dm1occ * 2
    rdm1[nocc:,nocc:] = dm1vir * 2
    for i in range(nocc):
        rdm1[i,i] += 2
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    return rdm1

def make_rdm1(mp, t2, verbose=logger.NOTE):
    '''1-particle density matrix in MO basis.  The off-diagonal blocks due to
    the orbital response contribution are not included.
    '''
    if isinstance(verbose, numpy.ndarray):
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v1.0-alpha.
The old make_rdm1 has been renamed to make_rdm1_ao.
Given t2 amplitude, current function returns 1-RDM in MO basis''')
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    dm1occ = numpy.zeros((nocc,nocc))
    dm1vir = numpy.zeros((nvir,nvir))
    for i in range(nocc):
        dm1vir += numpy.einsum('jca,jcb->ab', t2[i], t2[i]) * 2 \
                - numpy.einsum('jca,jbc->ab', t2[i], t2[i])
        dm1occ += numpy.einsum('iab,jab->ij', t2[i], t2[i]) * 2 \
                - numpy.einsum('iab,jba->ij', t2[i], t2[i])
    rdm1 = numpy.zeros((nmo,nmo))
# *2 for beta electron
    rdm1[:nocc,:nocc] =-dm1occ * 2
    rdm1[nocc:,nocc:] = dm1vir * 2
    for i in range(nocc):
        rdm1[i,i] += 2
    return rdm1


def make_rdm2(mp, t2, verbose=logger.NOTE):
    '''2-RDM in MO basis'''
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo)) # Chemist notation
    #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
    #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
    for i in range(nocc):
        t2i = t2[i]
        dm2[i,nocc:,:nocc,nocc:] = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
        dm2[nocc:,i,nocc:,:nocc] = dm2[i,nocc:,:nocc,nocc:].transpose(0,2,1)

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2
    return dm2


class MP2(lib.StreamObject):
    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nocc = self.mol.nelectron // 2
        self.nmo = len(mf.mo_energy)

        self.emp2 = None
        self.e_corr = None
        self.t2 = None

    def kernel(self, mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if mo_coeff is None:
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need mf.kernel() to generate them.')
            raise RuntimeError

        self.emp2, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
        logger.log(self, 'RMP2 energy = %.15g', self.emp2)
        self.e_corr = self.emp2
        return self.emp2, self.t2

    # return eri_ovov array[nocc*nvir,nocc*nvir]
    def ao2mo(self, mo_coeff):
        log = logger.Logger(self.stdout, self.verbose)
        time0 = (time.clock(), time.time())
        log.debug('transform (ia|jb)')
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        co = mo_coeff[:,:nocc]
        cv = mo_coeff[:,nocc:]
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]
        if mem_now < mem_basic:
            warnings.warn('%s: Not enough memory. Available mem %s MB, required mem %s MB\n' %
                          (self.ao2mo, mem_now, mem_basic))
        if hasattr(self._scf, 'with_df') and self._scf.with_df:
            # To handle the PBC or custom 2-electron with 3-index tensor.
            # Call dfmp2.MP2 for efficient DF-MP2 implementation.
            log.warn('MP2 detected DF being bound to the HF object. '
                     '(ia|jb) is computed based on the DF 3-tensor integrals.\n'
                     'You can switch to dfmp2.MP2 for the DF-MP2 implementation')
            eri = self._scf.with_df.ao2mo((co,cv,co,cv))
        elif (self._scf._eri is not None and
            mem_incore+mem_now < self.max_memory or
            self.mol.incore_anyway):
            if self._scf._eri is None:
                eri = self.intor('int2e', aosym='s8')
            else:
                eri = self._scf._eri
            eri = ao2mo.incore.general(eri, (co,cv,co,cv))
        else:
            max_memory = max(2000, self.max_memory*.9-mem_now)
            erifile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo.outcore.general(self.mol, (co,cv,co,cv), erifile.name,
                                  max_memory=max_memory, verbose=self.verbose)
            eri = erifile
        time1 = log.timer('Integral transformation', *time0)
        return ao2mo.load(eri)

    def make_rdm1(self, t2=None):
        if t2 is None: t2 = self.t2
        return make_rdm1(self, t2, self.verbose)

    def make_rdm2(self, t2=None):
        if t2 is None: t2 = self.t2
        return make_rdm2(self, t2, self.verbose)

def _mem_usage(nocc, nvir):
    nmo = nocc + nvir
    basic = ((nocc*nvir)**2 + nocc*nvir**2*2)*8 / 1e6
    incore = nocc*nvir*nmo**2/2*8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic


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

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    print(mf.scf())

    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc

    co = mf.mo_coeff[:,:nocc]
    cv = mf.mo_coeff[:,nocc:]
    g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
    eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
    t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
    t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)

    pt = MP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204019967288338)
    print('incore', numpy.allclose(t2, t2ref0))
    pt.max_memory = 1
    print('direct', numpy.allclose(pt.kernel()[1], t2ref0))

    rdm1 = make_rdm1_ao(pt, mf.mo_energy, mf.mo_coeff)
    print(numpy.allclose(reduce(numpy.dot, (mf.mo_coeff, pt.make_rdm1(),
                                            mf.mo_coeff.T)), rdm1))

    eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
    rdm2 = pt.make_rdm2()
    e1 = numpy.einsum('ij,ij', mf.make_rdm1(), mf.get_hcore())
    e2 = .5 * numpy.dot(eri.flatten(), rdm2.flatten())
    print(e1+e2+mf.energy_nuc()-mf.e_tot - -0.204019976381)

    pt = MP2(scf.density_fit(mf))
    print(pt.kernel()[0] - -0.204254500454)
