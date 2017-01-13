#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Gaussian and planewaves mixed density fitting
Ref:
'''

import time
import copy
import tempfile
import ctypes
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
import pyscf.df
import pyscf.df.mdf
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df import mdf_jk
from pyscf.pbc.df import mdf_ao2mo
from pyscf.pbc.df.df import estimate_eta, make_modrho_basis, fuse_auxcell, \
        unique, _load3c
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, is_zero, gamma_point


class MDF(df.DF):
    '''Gaussian and planewaves mixed density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.metric = 'T'  # or 'S'
        # approximate short range fitting level
        # 0: no approximation;  1: (FIXME) gamma point for k-points;
        # 2: non-pbc approximation;  3: atomic approximation
        self.approx_sr_level = 0
        self.charge_constraint = False  # To adpat to pyscf.df.mdf module
        df.DF.__init__(self, cell, kpts)

    def dump_flags(self):
        df.DF.dump_flags(self)
        logger.info(self, 'metric = %s', self.metric)
        logger.info(self, 'approx_sr_level = %s', self.approx_sr_level)

    def build(self, j_only=False, with_j3c=True):
        log = logger.Logger(self.stdout, self.verbose)
        t1 = (time.clock(), time.time())
        self.dump_flags()

        self.auxcell = make_modrho_basis(self.cell, self.auxbasis, self.eta)

        self._j_only = j_only
        if j_only:
            kptij_lst = numpy.hstack((self.kpts,self.kpts)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, self.kpts[j])
                         for i, ki in enumerate(self.kpts) for j in range(i+1)]
            kptij_lst = numpy.asarray(kptij_lst)

        if not isinstance(self._cderi, str):
            if isinstance(self._cderi_file, str):
                self._cderi = self._cderi_file
            else:
                self._cderi = self._cderi_file.name

        if with_j3c:
            if self.approx_sr_level == 0:
                build_Lpq_pbc(self, self.auxcell, kptij_lst)
            elif self.approx_sr_level == 1:
                build_Lpq_pbc(self, self.auxcell, numpy.zeros((1,2,3)))
            elif self.approx_sr_level == 2:
                build_Lpq_nonpbc(self, self.auxcell)
            elif self.approx_sr_level == 3:
                build_Lpq_1c_approx(self, self.auxcell)
            t1 = log.timer_debug1('Lpq', *t1)

            _make_j3c(self, self.cell, self.auxcell, kptij_lst)
            t1 = log.timer_debug1('j3c', *t1)
        return self

    def load_Lpq(self, kpti_kptj=numpy.zeros((2,3))):
        with h5py.File(self._cderi, 'r') as f:
            if self.approx_sr_level == 0:
                return _load3c(self._cderi, 'Lpq', kpti_kptj)
            else:
                kpti, kptj = kpti_kptj
                if is_zero(kpti-kptj):
                    return pyscf.df.addons.load(self._cderi, 'Lpq/0')
                else:
                    # See _fake_Lpq_kpts
                    return pyscf.df.addons.load(self._cderi, 'Lpq/1')

    def load_j3c(self, kpti_kptj=numpy.zeros((2,3))):
        return _load3c(self._cderi, 'j3c', kpti_kptj)

    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None):
        '''Short range part'''
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        is_real = is_zero(kpti_kptj)
        nao = self.cell.nao_nr()
        if blksize is None:
            if is_real:
                if unpack:
                    blksize = max_memory*1e6/8/(nao*(nao+1)//2+nao**2*2)
                else:
                    blksize = max_memory*1e6/8/(nao*(nao+1)*2)
            else:
                blksize = max_memory*1e6/16/(nao**2*3)
            blksize = max(16, min(int(blksize), self.blockdim))
            logger.debug2(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

        if unpack:
            buf = numpy.empty((blksize,nao*(nao+1)//2))
        def load(Lpq, b0, b1, bufR, bufI):
            Lpq = numpy.asarray(Lpq[b0:b1])
            if is_real:
                if unpack:
                    LpqR = lib.unpack_tril(Lpq, out=bufR).reshape(-1,nao**2)
                else:
                    LpqR = Lpq
                LpqI = numpy.zeros_like(LpqR)
            else:
                shape = Lpq.shape
                if unpack:
                    tmp = numpy.ndarray(shape, buffer=buf)
                    tmp[:] = Lpq.real
                    LpqR = lib.unpack_tril(tmp, out=bufR).reshape(-1,nao**2)
                    tmp[:] = Lpq.imag
                    LpqI = lib.unpack_tril(tmp, lib.ANTIHERMI, out=bufI).reshape(-1,nao**2)
                else:
                    LpqR = numpy.ndarray(shape, buffer=bufR)
                    LpqR[:] = Lpq.real
                    LpqI = numpy.ndarray(shape, buffer=bufI)
                    LpqI[:] = Lpq.imag
            return LpqR, LpqI

        LpqR = LpqI = j3cR = j3cI = None
        with self.load_Lpq(kpti_kptj) as Lpq:
            naux = Lpq.shape[0]
            with self.load_j3c(kpti_kptj) as j3c:
                for b0, b1 in lib.prange(0, naux, blksize):
                    LpqR, LpqI = load(Lpq, b0, b1, LpqR, LpqI)
                    j3cR, j3cI = load(j3c, b0, b1, j3cR, j3cI)
                    yield LpqR, LpqI, j3cR, j3cI

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = mdf_ao2mo.get_eri
    ao2mo = get_mo_eri = mdf_ao2mo.general

    def update_mp(self):
        pass

    def update_cc(self):
        pass

    def update(self):
        pass

################################################################################
# With this function to mimic the molecular DF.loop function, the pbc gamma
# point DF object can be used in the molecular code
    def loop(self):
        raise RuntimeError('MDF method does not support the symmetric-DF interface')


def build_Lpq_pbc(mydf, auxcell, kptij_lst):
    '''Fitting coefficients for auxiliary functions'''
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpts_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpts_ji)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    if mydf.metric.upper() == 'S':
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_sph',
                       kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=max_memory)
        s_aux = auxcell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=uniq_kpts)
    elif mydf.metric.upper() == 'T':
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_p2_sph',
                       kptij_lst=kptij_lst, dataname='Lpq', max_memory=max_memory)
        s_aux = [x*2 for x in auxcell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=uniq_kpts)]
    elif mydf.metric.upper() == 'J':
        fused_cell, fuse = fuse_auxcell(mydf, auxcell)
        outcore.aux_e2(mydf.cell, fused_cell, mydf._cderi, 'cint3c2e_sph',
                       kptij_lst=kptij_lst, dataname='j3c', max_memory=max_memory)
        vbar = fuse(mydf.auxbar(fused_cell))
        with h5py.File(mydf._cderi) as f:
            f['Lpq-kptij'] = kptij_lst
            for k_uniq, kpt_uniq in enumerate(uniq_kpts):
                adapted_ji_idx = numpy.where(uniq_inverse == k_uniq)[0]
                adapted_kptjs = kptjs[adapted_ji_idx]
                if is_zero(kpt_uniq):
                    ovlp = mydf.cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=adapted_kptjs)
                    for k, ji in enumerate(adapted_ji_idx):
                        ovlp[k] = lib.pack_tril(ovlp[k])
                for k, idx in enumerate(adapted_ji_idx):
                    v = fuse(numpy.asarray(f['j3c/%d'%idx]))
                    if is_zero(kpt_uniq):
                        for i, c in enumerate(vbar):
                            if c != 0:
                                v[i] -= c * ovlp[k]
                    f['Lpq/%d'%idx] = v
        v = ovlp = vbar = None

        j2c = fused_cell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
        for k, kpt in enumerate(uniq_kpts):
            j2c[k] = fuse(fuse(j2c[k]).T).T.copy()
        s_aux = j2c

#    else: # T+S
#        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_sph',
#                       kptij_lst=kptij_lst, dataname='Lpq_s',
#                       max_memory=max_memory)
#        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_p2_sph',
#                       kptij_lst=kptij_lst, dataname='Lpq',
#                       max_memory=max_memory)
#        with h5py.File(mydf._cderi) as f:
#            for k in range(len(kptij_lst)):
#                f['Lpq/%d'%k][:] = f['Lpq/%d'%k].value + f['Lpq_s/%d'%k].value
#                del(f['Lpq_s/%d'%k])
#        s_aux = auxcell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=uniq_kpts)
#        s_aux = [x+y*2 for x,y in zip(s_aux, auxcell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=uniq_kpts))]

    try:
        s_aux = [scipy.linalg.cho_factor(x) for x in s_aux]
    except scipy.linalg.LinAlgError:
        eigs = [scipy.linalg.eigh(x)[0] for x in s_aux]
        conds = [x[-1]/max(1e-16, x[0]) for x in eigs]
        n = eigs[0].size
        shift = [0] * len(s_aux)
        for i, x in enumerate(s_aux):
            if conds[i] > 1e15:
                shift[i] = max(abs(eigs[i][0])*2, eigs[i][-1]*1e-18)
                x += numpy.eye(n) * shift[i]
        logger.warn(mydf, 'Ill condition number %s found in metric %s.\n'
                    'Level shift %s is applied.',
                    conds, mydf.metric, shift)
        s_aux = [scipy.linalg.cho_factor(x) for x in s_aux]

    max_memory = mydf.max_memory - lib.current_memory()[0]
    naux = auxcell.nao_nr()
    blksize = max(int(max_memory*.5*1e6/16/naux/mydf.blockdim), 1) * mydf.blockdim
    with h5py.File(mydf._cderi) as feri:
        for k, where in enumerate(uniq_inverse):
            s_k = s_aux[where]
            key = 'Lpq/%d' % k
            Lpq = feri[key]
            nao_pair = Lpq.shape[1]
            for p0, p1 in lib.prange(0, nao_pair, blksize):
                Lpq[:,p0:p1] = scipy.linalg.cho_solve(s_k, Lpq[:,p0:p1])

# TODO: replace incore with outcore
def build_Lpq_nonpbc(mydf, auxcell):
    if mydf.metric.upper() == 'S':
        j3c = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_ovlp_sph')
    else:  # mydf.metric.upper() == 'T'
        j3c = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_p2_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_kin_sph') * 2

    naux = auxcell.nao_nr()
    nao = mydf.cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao_pair)) # 512K
        Lpq = feri.create_dataset('Lpq/0', (naux,nao_pair), 'f8', chunks=chunks)
        Lpq[:] = lib.cho_solve(j2c, j3c.T)

def build_Lpq_1c_approx(mydf, auxcell):
    get_Lpq = pyscf.df.mdf._make_Lpq_atomic_approx(mydf, mydf.cell, auxcell)

    max_memory = mydf.max_memory - lib.current_memory()[0]
    naux = auxcell.nao_nr()
    nao = mydf.cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = max(int(max_memory*.5*1e6/8/naux/mydf.blockdim), 1) * mydf.blockdim
    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao_pair)) # 512K
        Lpq = feri.create_dataset('Lpq/0', (naux,nao_pair), 'f8', chunks=chunks)
        for p0, p1 in lib.prange(0, nao_pair, blksize):
            v = get_Lpq(None, p0, p1)
            Lpq[:,p0:p1] = get_Lpq(None, p0, p1)


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell(mydf, mydf.auxcell)
    if mydf.metric.upper() != 'J':
        outcore.aux_e2(cell, fused_cell, mydf._cderi, 'cint3c2e_sph',
                       kptij_lst=kptij_lst, dataname='j3c', max_memory=max_memory)
    t1 = log.timer_debug1('3c2e', *t1)

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    gs = mydf.gs
    Gv, Gvbase, kws = cell.get_Gv_weights(gs)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngs = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
    kLRs = []
    kLIs = []
    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        aoaux = fuse(aoaux)
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, gs))
        kLR = (aoaux.real * coulG).T
        kLI = (aoaux.imag * coulG).T
        if not kLR.flags.c_contiguous: kLR = lib.transpose(kLR.T)
        if not kLI.flags.c_contiguous: kLI = lib.transpose(kLI.T)

        j2c[k] = fuse(fuse(j2c[k]).T).T.copy()
        if is_zero(kpt):  # kpti == kptj
            j2c[k] -= lib.dot(kLR.T, kLR)
            j2c[k] -= lib.dot(kLI.T, kLI)
        else:
             # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
            j2cR, j2cI = zdotCN(kLR.T, kLI.T, kLR, kLI)
            j2c[k] -= j2cR + j2cI * 1j

        kLR *= coulG.reshape(-1,1)
        kLI *= coulG.reshape(-1,1)
        kLRs.append(kLR)
        kLIs.append(kLI)
        aoaux = kLR = kLI = j2cR = j2cI = coulG = None

    feri = h5py.File(mydf._cderi)
    log.debug2('memory = %s', lib.current_memory()[0])

    # Expand approx Lpq for aosym='s1'.  The approx Lpq are all in aosym='s2' mode
    if mydf.approx_sr_level > 0 and len(kptij_lst) > 1:
        Lpq_fake = _fake_Lpq_kpts(mydf, feri, naux, nao)

    def save(label, dat, col0, col1):
        nrow = dat.shape[0]
        feri[label][:nrow,col0:col1] = dat

    def make_kpt(uniq_kptji_id):  # kpt = kptj - kpti
        kpt = uniq_kpts[uniq_kptji_id]
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
        kLR = kLRs[uniq_kptji_id]
        kLI = kLIs[uniq_kptji_id]

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=adapted_kptjs)
            for k, ji in enumerate(adapted_ji_idx):
                ovlp[k] = lib.pack_tril(ovlp[k])
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.6*1e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = pyscf.df.outcore._guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.2*1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.4*1e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngs, 16384)
        pqkRbuf = numpy.empty(buflen*Gblksize)
        pqkIbuf = numpy.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = numpy.zeros((nkptj,buflen*Gblksize), dtype=numpy.complex128)

        col1 = 0
        for istep, sh_range in enumerate(shranges):
            log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d', \
                       istep+1, len(shranges), *sh_range)
            bstart, bend, ncol = sh_range
            col0, col1 = col1, col1+ncol
            j3cR = []
            j3cI = []
            for k, idx in enumerate(adapted_ji_idx):
                v = fuse(numpy.asarray(feri['j3c/%d'%idx][:,col0:col1]))

                if mydf.approx_sr_level == 0:
                    Lpq = numpy.asarray(feri['Lpq/%d'%idx][:,col0:col1])
                elif aosym == 's2':
                    Lpq = numpy.asarray(feri['Lpq/0'][:,col0:col1])
                else:
                    Lpq = numpy.asarray(Lpq_fake[:,col0:col1])
                lib.dot(j2c[uniq_kptji_id], Lpq, -.5, v, 1)
                if is_zero(kpt):
                    for i, c in enumerate(vbar):
                        if c != 0:
                            v[i] -= c * ovlp[k][col0:col1]

                j3cR.append(numpy.asarray(v.real, order='C'))
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    j3cI.append(None)
                else:
                    j3cI.append(numpy.asarray(v.imag, order='C'))
            v = Lpq = None
            log.debug3('  istep, k = %d %d  memory = %s',
                       istep, k, lib.current_memory()[0])

            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
                for p0, p1 in lib.prange(0, ngs, Gblksize):
                    ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                          b, gxyz[p0:p1], Gvbase, kpt,
                                          adapted_kptjs, out=buf)
                    nG = p1 - p0
                    for k, ji in enumerate(adapted_ji_idx):
                        aoao = numpy.ndarray((nG,ncol), dtype=numpy.complex128,
                                             order='F', buffer=buf[k])
                        pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                        pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                        pqkR[:] = aoao.real.T
                        pqkI[:] = aoao.imag.T
                        aoao[:] = 0
                        lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k], 1)
                        lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k], 1)
                        if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                            lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k], 1)
                            lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k], 1)
                    log.debug3('  p0:p1 = %d:%d  memory = %s',
                               p0, p1, lib.current_memory()[0])
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)
                ni = ncol // nao
                for p0, p1 in lib.prange(0, ngs, Gblksize):
                    ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                          b, gxyz[p0:p1], Gvbase, kpt,
                                          adapted_kptjs, out=buf)
                    nG = p1 - p0
                    for k, ji in enumerate(adapted_ji_idx):
                        aoao = numpy.ndarray((nG,ni,nao), dtype=numpy.complex128,
                                             order='F', buffer=buf[k])
                        pqkR = numpy.ndarray((ni,nao,nG), buffer=pqkRbuf)
                        pqkI = numpy.ndarray((ni,nao,nG), buffer=pqkIbuf)
                        pqkR[:] = aoao.real.transpose(1,2,0)
                        pqkI[:] = aoao.imag.transpose(1,2,0)
                        aoao[:] = 0
                        pqkR = pqkR.reshape(-1,nG)
                        pqkI = pqkI.reshape(-1,nG)
                        zdotCN(kLR[p0:p1].T, kLI[p0:p1].T, pqkR.T, pqkI.T,
                               -1, j3cR[k], j3cI[k], 1)
                    log.debug3('  p0:p1 = %d:%d  memory = %s',
                               p0, p1, lib.current_memory()[0])

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    save('j3c/%d'%ji, j3cR[k], col0, col1)
                else:
                    save('j3c/%d'%ji, j3cR[k]+j3cI[k]*1j, col0, col1)


    for k, kpt in enumerate(uniq_kpts):
        make_kpt(k)

    feri.close()

def _fake_Lpq_kpts(mydf, feri, naux, nao):
    chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao**2)) # 512K
    Lpq = feri.create_dataset('Lpq/1', (naux,nao**2), 'f8', chunks=chunks)
    for p0, p1 in lib.prange(0, naux, mydf.blockdim):
        Lpq[p0:p1] = lib.unpack_tril(feri['Lpq/0'][p0:p1]).reshape(-1,nao**2)
    return Lpq
