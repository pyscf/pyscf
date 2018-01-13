#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
from functools import reduce
import copy
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo


'''
spin-adapted MP2
t2[i,j,a,b] = (ia|jb) / D_ij^ab
'''

def kernel(mp, mo_energy, mo_coeff, eris=None, with_t2=True,
           verbose=logger.NOTE):
    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    mo_e = _mo_energy_without_core(mp, mo_energy)
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None

    emp2 = 0.0
    ovov = eris.ovov
    for i in range(nocc):
        gi = numpy.asarray(ovov[i*nvir:(i+1)*nvir])
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
        # 2*ijab-ijba
        theta = gi*2 - gi.transpose(0,2,1)
        emp2 += numpy.einsum('jab,jab', t2i, theta)
        #emp2 -= numpy.einsum('jab,jab', t2i, theta)
        if with_t2:
            t2[i] = t2i

    return emp2, t2

# Need less memory
def make_rdm1_ao(mp, mo_energy, mo_coeff, eris=None, verbose=logger.NOTE):
    mp = copy.copy(mp)
    mp.mo_energy = mo_energy
    mp.mo_coeff = mo_coeff
    rdm1_mo = make_rdm1(mp, None, eris, verbose)
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1_mo, mo_coeff.T))
    return rdm1

def make_rdm1(mp, t2=None, eris=None, verbose=logger.NOTE):
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
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo(mp.mo_coeff)
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    dm1occ = numpy.zeros((nocc,nocc))
    dm1vir = numpy.zeros((nvir,nvir))
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
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
    return rdm1


def make_rdm2(mp, t2, eris=None, verbose=logger.NOTE):
    '''2-RDM in MO basis'''
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo(mp.mo_coeff)
        mo_energy = _mo_energy_without_core(mp, mp.mo_energy)
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo)) # Chemist notation
    #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
    #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        dm2[i,nocc:,:nocc,nocc:] = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
        dm2[nocc:,i,nocc:,:nocc] = dm2[i,nocc:,:nocc,nocc:].transpose(0,2,1)

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2
    return dm2


class MP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.emp2 = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        elif isinstance(self.frozen, (int, numpy.integer)):
            return int(self.mo_occ.sum()) // 2 - self.frozen
        elif self.frozen:
            occ_idx = self.mo_occ > 0
            occ_idx[numpy.asarray(self.frozen)] = False
            return numpy.count_nonzero(occ_idx)
        else:
            return int(self.mo_occ.sum()) // 2
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        if isinstance(self.frozen, (int, numpy.integer)):
            return len(self.mo_occ) - self.frozen
        else:
            return len(self.mo_occ) - len(self.frozen)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=True):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.emp2, self.t2 = \
                kernel(self, mo_energy, mo_coeff, eris, with_t2, verbose=self.verbose)
        logger.log(self, 'RMP2 energy = %.15g', self.emp2)
        self.e_corr = self.emp2
        return self.emp2, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff, verbose=self.verbose)

    def make_rdm1(self, t2=None, eris=None):
        if t2 is None: t2 = self.t2
        return make_rdm1(self, t2, eris, verbose=self.verbose)

    def make_rdm2(self, t2=None, eris=None):
        if t2 is None: t2 = self.t2
        return make_rdm2(self, t2, eris, verbose=self.verbose)

def _mo_energy_without_core(mp, mo_energy):
    return mo_energy[_active_idx(mp)]

def _mo_without_core(mp, mo):
    return mo[:,_active_idx(mp)]

def _active_idx(mp):
    moidx = numpy.ones(mp.mo_occ.size, dtype=numpy.bool)
    if isinstance(mp.frozen, (int, numpy.integer)):
        moidx[:mp.frozen] = False
    elif len(mp.frozen) > 0:
        moidx[numpy.asarray(mp.frozen)] = False
    return moidx

def _mem_usage(nocc, nvir):
    nmo = nocc + nvir
    basic = ((nocc*nvir)**2 + nocc*nvir**2*2)*8 / 1e6
    incore = nocc*nvir*nmo**2/2*8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

class _ERIS:
    def __init__(self, mp, mo_coeff=None, verbose=None):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = _mo_without_core(mp, mp.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = _mo_without_core(mp, mo_coeff)

        nocc = mp.nocc
        nmo = mp.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mp.max_memory*.9-mem_now)
        log = logger.Logger(mp.stdout, mp.verbose)
        if max_memory < mem_basic:
            log.warn('Not enough memory for integral transformation. '
                     'Available mem %s MB, required mem %s MB',
                     max_memory, mem_basic)

        time0 = (time.clock(), time.time())

        co = numpy.asarray(mo_coeff[:,:nocc], order='F')
        cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
        if hasattr(mp._scf, 'with_df') and mp._scf.with_df:
            # To handle the PBC or custom 2-electron with 3-index tensor.
            # Call dfmp2.MP2 for efficient DF-MP2 implementation.
            log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                     '3-tensor integrals.\n'
                     'You can switch to dfmp2.MP2 for better performance')
            log.debug('transform (ia|jb) with_df')
            self.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

        elif (mp.mol.incore_anyway or
              (mp._scf._eri is not None and
               mem_incore+mem_now < mp.max_memory)):
            log.debug('transform (ia|jb) incore')
            self.ovov = ao2mo.incore.general(mp._scf._eri, (co,cv,co,cv))

        else:
            log.debug('transform (ia|jb) outcore')
            self.feri = lib.H5TmpFile()
            #ao2mo.outcore.general(mp.mol, (co,cv,co,cv), self.feri,
            #                      max_memory=max_memory, verbose=mp.verbose)
            #self.ovov = self.feri['eri_mo']
            self.ovov = _ao2mo_ovov(mp, co, cv, self.feri, max_memory, mp.verbose)

        time1 = log.timer('Integral transformation', *time0)

#
# the MO integral for MP2 is (ov|ov). This is the efficient integral
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)
#
def _ao2mo_ovov(mp, orbo, orbv, feri, max_memory=2000, verbose=None):
    time0 = (time.clock(), time.time())
    log = logger.new_logger(mp, verbose)

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    nbas = mol.nbas
    assert(nvir <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc**2*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    buf_i = numpy.empty((nocc*dmax**2*nao))
    buf_li = numpy.empty((nocc**2*dmax**2))
    buf1 = numpy.empty_like(buf_li)

    fint = gto.moleintor.getints4c
    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = numpy.ndarray((nocc,(i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc,nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T, tmp_i.reshape(nocc*(i1-i0)*(j1-j0),nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc,nocc,(i1-i0),(j1-j0))
                save(str(count), tmp_li.transpose(1,0,2,3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    chunks = (nvir,nvir)
    h5dat = feri.create_dataset('ovov', (nocc*nvir,nocc*nvir), 'f8',
                                chunks=chunks)
    # jk_where is the sorting indices for the stacked (oO|pP) integrals in pass 2
    jk_where = []
    aoao_idx = numpy.arange(nao*nao).reshape(nao,nao)
    for i0, i1, j0, j1 in jk_blk_slices:
        # idx of pP in <oO|pP>
        jk_where.append(aoao_idx[i0:i1,j0:j1].ravel())
        if i0 != j0:
            # idx of pP in (<oO|pP>).transpose(1,0,3,2)
            jk_where.append(aoao_idx[j0:j1,i0:i1].ravel())
    jk_where = numpy.argsort(numpy.hstack(jk_where)).astype(numpy.int32)
    orbv = numpy.asarray(orbv, order='F')

    occblk = int(min(nocc, max(4, 250/nocc, max_memory*.9e6/8/(nao**2*nocc)/5)))
    def load(i0, eri):
        if i0 >= nocc:
            return
        i1 = min(i0+occblk, nocc)
        eri = eri[:(i1-i0)*nocc]
        p1 = 0
        for k, jk_slice in enumerate(jk_blk_slices):
            dat = numpy.asarray(ftmp[str(k)][i0:i1]).reshape((i1-i0)*nocc,-1)
            p0, p1 = p1, p1 + dat.shape[1]
            eri[:,p0:p1] = dat
            if jk_slice[0] != jk_slice[2]:
                dat = numpy.asarray(ftmp[str(k)][:,i0:i1])
                dat = dat.transpose(1,0,3,2).reshape((i1-i0)*nocc,-1)
                p0, p1 = p1, p1 + dat.shape[1]
                eri[:,p0:p1] = dat

    def save(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,nocc*nvir)

    buf_prefecth = numpy.empty((occblk*nocc,nao**2))
    buf = numpy.empty_like(buf_prefecth)
    buf1 = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                eri = buf[:(i1-i0)*nocc]
                prefetch(i1, buf_prefecth)

                idx = numpy.arange(eri.shape[0], dtype=numpy.int32)
                dat = lib.take_2d(eri, idx, jk_where, out=buf1)
                dat = _ao2mo.nr_e2(dat, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    return h5dat

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol).run()
    mp = MP2(mf)
    mp.verbose = 5
    #print mp.kernel(with_t2=False)

############
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

    pt = MP2(scf.density_fit(mf, 'weigend'))
    print(pt.kernel()[0] - -0.204254500454)
