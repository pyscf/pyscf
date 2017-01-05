#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.

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
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.df.mdf import _uncontract_basis
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df import pwdf
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.df.pwdf import estimate_eta, get_nuc

KPT_DIFF_TOL = 1e-6

def make_modrho_basis(cell, auxbasis=None, drop_eta=1.):
    auxcell = copy.copy(cell)
    if auxbasis is None:
        _basis = _uncontract_basis(cell, auxbasis)
    elif isinstance(auxbasis, str):
        uniq_atoms = set([a[0] for a in cell._atom])
        _basis = auxcell.format_basis(dict([(a, auxbasis) for a in uniq_atoms]))
    else:
        _basis = auxcell.format_basis(auxbasis)
    auxcell._basis = _basis
    auxcell._atm, auxcell._bas, auxcell._env = \
            auxcell.make_env(cell._atom, auxcell._basis, cell._env[:gto.PTR_ENV_START])

# Note libcint library will multiply the norm of the integration over spheric
# part sqrt(4pi) to the basis.
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    steep_shls = []
    ndrop = 0
    for ib in range(len(auxcell._bas)):
        l = auxcell.bas_angular(ib)
        np = auxcell.bas_nprim(ib)
        nc = auxcell.bas_nctr(ib)
        es = auxcell.bas_exp(ib)
        ptr = auxcell._bas[ib,gto.PTR_COEFF]
        cs = auxcell._env[ptr:ptr+np*nc].reshape(nc,np).T

        if numpy.any(es < drop_eta):
            cs = cs[es>=drop_eta]
            es = es[es>=drop_eta]
            np, ndrop = len(es), ndrop+np-len(es)
            pe = auxcell._bas[ib,gto.PTR_EXP]
            auxcell._bas[ib,gto.NPRIM_OF] = np
            auxcell._env[pe:pe+np] = es

        if np > 0:
# int1 is the multipole value. l*2+2 is due to the radial part integral
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
            int1 = gto.mole._gaussian_int(l*2+2, es)
            s = numpy.einsum('pi,p->i', cs, int1)
# The auxiliary basis normalization factor is not a must for density expansion.
# half_sph_norm here to normalize the monopole (charge).  This convention can
# simplify the formulism of \int \bar{\rho}, see function auxbar.
            cs = numpy.einsum('pi,i->pi', cs, half_sph_norm/s)
            auxcell._env[ptr:ptr+np*nc] = cs.T.reshape(-1)
            steep_shls.append(ib)

    auxcell._bas = numpy.asarray(auxcell._bas[steep_shls], order='C')
    auxcell._built = True
    logger.debug(cell, 'Drop %d primitive fitting functions', ndrop)
    logger.debug(cell, 'make aux basis, num shells = %d, num cGTOs = %d',
                 auxcell.nbas, auxcell.nao_nr())
    return auxcell

def make_modchg_basis(auxcell, smooth_eta, l_max=3):
# * chgcell defines smooth gaussian functions for each angular momentum for
#   auxcell. The smooth functions may be used to carry the charge
    chgcell = copy.copy(auxcell)  # smooth model density for coulomb integral to carry charge
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    chg_bas = []
    chg_env = [smooth_eta]
    ptr_eta = auxcell._env.size
    ptr = ptr_eta + 1
    for ia in range(auxcell.natm):
        for l in set(auxcell._bas[auxcell._bas[:,gto.ATOM_OF]==ia, gto.ANG_OF]):
            if l <= l_max:
                norm = half_sph_norm/gto.mole._gaussian_int(l*2+2, smooth_eta)
                chg_bas.append([ia, l, 1, 1, 0, ptr_eta, ptr, 0])
                chg_env.append(norm)
                ptr += 1

    chgcell._atm = auxcell._atm
    chgcell._bas = numpy.asarray(chg_bas, dtype=numpy.int32).reshape(-1,gto.BAS_SLOTS)
    chgcell._env = numpy.hstack((auxcell._env, chg_env))
    chgcell._built = True
    logger.debug1(auxcell, 'make smooth basis, num shells = %d, num cGTOs = %d',
                  chgcell.nbas, chgcell.nao_nr())
    return chgcell


class DF(pwdf.PWDF):
    '''Gaussian and planewaves mixed density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.gs = cell.gs
        self.auxbasis = None
        self.eta = estimate_eta(cell)

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.blockdim = 256
        self._j_only = False
        self._cderi_file = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self._cderi = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'auxbasis = %s', self.auxbasis)
        logger.info(self, 'eta = %s', self.eta)
        if isinstance(self._cderi, str):
            logger.info(self, '_cderi = %s', self._cderi)
        else:
            logger.info(self, '_cderi = %s', self._cderi_file.name)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)

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
            _make_j3c(self, self.cell, self.auxcell, kptij_lst)
            t1 = log.timer_debug1('j3c', *t1)
        return self

    def auxbar(self, fused_cell=None):
        '''
        Potential average = \sum_L V_L*Lpq

        The coulomb energy is computed with chargeless density
        \int (rho-C) V,  C = (\int rho) / vol = Tr(gamma,S)/vol
        It is equivalent to removing the averaged potential from the short range V
        vs = vs - (\int V)/vol * S
        '''
        if fused_cell is None:
            fused_cell, fuse = fuse_auxcell(self, self.auxcell)
        aux_loc = fused_cell.ao_loc_nr()
        vbar = numpy.zeros(aux_loc[-1])
        if fused_cell.dimension != 3:
            return vbar

        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        for i in range(fused_cell.nbas):
            l = fused_cell.bas_angular(i)
            if l == 0:
                es = fused_cell.bas_exp(i)
                if es.size == 1:
                    vbar[aux_loc[i]] = -1/es[0]
                else:
# Remove the normalization to get the primitive contraction coeffcients
                    norms = half_sph_norm/gto.mole._gaussian_int(2, es)
                    cs = numpy.einsum('i,ij->ij', 1/norms, fused_cell._libcint_ctr_coeff(i))
                    vbar[aux_loc[i]:aux_loc[i+1]] = numpy.einsum('in,i->n', cs, -1/es)
        vbar *= numpy.pi/fused_cell.vol
        return vbar

    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True):
        '''Short range part'''
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        is_real = is_zero(kpti_kptj)
        nao = self.cell.nao_nr()
        if is_real:
            if unpack:
                blksize = max_memory*1e6/8/(nao*(nao+1)//2+nao**2)
            else:
                blksize = max_memory*1e6/8/(nao*(nao+1))
        else:
            blksize = max_memory*1e6/16/(nao**2*2)
        blksize = max(16, min(int(blksize), self.blockdim))
        logger.debug3(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

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

        LpqR = LpqI = None
        with _load3c(self._cderi, 'j3c', kpti_kptj) as j3c:
            naux = j3c.shape[0]
            for b0, b1 in lib.prange(0, naux, blksize):
                LpqR, LpqI = load(j3c, b0, b1, LpqR, LpqI)
                yield LpqR, LpqI

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
            return df_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = df_ao2mo.get_eri
    ao2mo = get_mo_eri = df_ao2mo.general

    def update_mp(self):
        pass

    def update_cc(self):
        pass

    def update(self):
        pass

def unique(kpts):
    kpts = numpy.asarray(kpts)
    nkpts = len(kpts)
    uniq_kpts = []
    uniq_index = []
    uniq_inverse = numpy.zeros(nkpts, dtype=int)
    seen = numpy.zeros(nkpts, dtype=bool)
    n = 0
    for i, kpt in enumerate(kpts):
        if not seen[i]:
            uniq_kpts.append(kpt)
            uniq_index.append(i)
            idx = abs(kpt-kpts).sum(axis=1) < 1e-6
            uniq_inverse[idx] = n
            seen[idx] = True
            n += 1
    return numpy.asarray(uniq_kpts), numpy.asarray(uniq_index), uniq_inverse

def fuse_auxcell(mydf, auxcell):
    chgcell = make_modchg_basis(auxcell, mydf.eta)
    fused_cell = copy.copy(auxcell)
    fused_cell._atm, fused_cell._bas, fused_cell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)

    aux_loc = auxcell.ao_loc_nr()
    naux = aux_loc[-1]
    modchg_offset = -numpy.ones((chgcell.natm,8), dtype=int)
    smooth_loc = chgcell.ao_loc_nr()
    for i in range(chgcell.nbas):
        ia = chgcell.bas_atom(i)
        l  = chgcell.bas_angular(i)
        modchg_offset[ia,l] = smooth_loc[i]

    def fuse(Lpq):
        Lpq, chgLpq = Lpq[:naux], Lpq[naux:]
        for i in range(auxcell.nbas):
            l  = auxcell.bas_angular(i)
            ia = auxcell.bas_atom(i)
            p0 = modchg_offset[ia,l]
            if p0 >= 0:
                nd = (aux_loc[i+1] - aux_loc[i]) // auxcell.bas_nctr(i)
                for i0, i1 in lib.prange(aux_loc[i], aux_loc[i+1], nd):
                    Lpq[i0:i1] -= chgLpq[p0:p0+nd]
        return Lpq
    return fused_cell, fuse


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)
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
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, gs))
        LkR = aoaux.real * coulG
        LkI = aoaux.imag * coulG

        if is_zero(kpt):  # kpti == kptj
            j2c[k][naux:] -= lib.ddot(LkR[naux:], LkR.T)
            j2c[k][naux:] -= lib.ddot(LkI[naux:], LkI.T)
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T
        else:
            j2cR, j2cI = zdotCN(LkR[naux:], LkI[naux:], LkR.T, LkI.T)
            j2c[k][naux:] -= j2cR + j2cI * 1j
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T.conj()
        #j2c[k] = fuse(fuse(j2c[k]).T).T.copy()
        j2c[k] = scipy.linalg.cholesky(fuse(fuse(j2c[k]).T).T, lower=True)
        kLR = LkR[naux:].T
        kLI = LkI[naux:].T
        if not kLR.flags.c_contiguous: kLR = lib.transpose(LkR[naux:])
        if not kLI.flags.c_contiguous: kLI = lib.transpose(LkI[naux:])
        kLR *= coulG.reshape(-1,1)
        kLI *= coulG.reshape(-1,1)
        kLRs.append(kLR)
        kLIs.append(kLI)
        aoaux = LkR = LkI = kLR = kLI = coulG = None

    feri = h5py.File(mydf._cderi)

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

        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.6*1e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
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
                v = numpy.asarray(feri['j3c/%d'%idx][:,col0:col1])
                if is_zero(kpt):
                    for i, c in enumerate(vbar):
                        if c != 0:
                            v[i] -= c * ovlp[k][col0:col1]
                j3cR.append(numpy.asarray(v.real, order='C'))
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    j3cI.append(None)
                else:
                    j3cI.append(numpy.asarray(v.imag, order='C'))

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
                        lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k][naux:], 1)
                        lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k][naux:], 1)
                        if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                            lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k][naux:], 1)
                            lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k][naux:], 1)
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
                               -1, j3cR[k][naux:], j3cI[k][naux:], 1)

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = fuse(j3cR[k])
                else:
                    v = fuse(j3cR[k] + j3cI[k] * 1j)

                v = scipy.linalg.solve_triangular(j2c[uniq_kptji_id], v,
                                                  lower=True, overwrite_b=True)
                feri['j3c/%d'%ji][:naux,col0:col1] = v

    for k, kpt in enumerate(uniq_kpts):
        make_kpt(k)

    for k, kptij in enumerate(kptij_lst):
        v = feri['j3c/%d'%k][:naux]
        del(feri['j3c/%d'%k])
        feri['j3c/%d'%k] = v

    feri.close()

def is_zero(kpt):
    return abs(numpy.asarray(kpt)).sum() < KPT_DIFF_TOL
gamma_point = is_zero

class _load3c(object):
    def __init__(self, cderi, label, kpti_kptj):
        self.cderi = cderi
        self.label = label
        self.kpti_kptj = kpti_kptj
        self.feri = None

    def __enter__(self):
        self.feri = h5py.File(self.cderi, 'r')
        kpti_kptj = numpy.asarray(self.kpti_kptj)
        kptij_lst = self.feri['%s-kptij'%self.label].value
        dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj))
        k_id = numpy.where(dk < 1e-6)[0]
        if len(k_id) > 0:
            dat = self.feri['%s/%d' % (self.label,k_id[0])]
        else:
            # swap ki,kj due to the hermiticity
            kptji = kpti_kptj[[1,0]]
            dk = numpy.einsum('kij->k', abs(kptij_lst-kptji))
            k_id = numpy.where(dk < 1e-6)[0]
            if len(k_id) == 0:
                raise RuntimeError('%s for kpts %s is not initialized.\n'
                                   'Reset attribute .kpts then call '
                                   '.build() to initialize %s.'
                                   % (self.label, kpti_kptj, self.label))
            dat = self.feri['%s/%d' % (self.label, k_id[0])]
            dat = _load_and_unpack(dat)
        return dat

    def __exit__(self, type, value, traceback):
        self.feri.close()

class _load_and_unpack(object):
    def __init__(self, dat):
        self.dat = dat
        self.shape = self.dat.shape
    def __getslice__(self, p0, p1):
        nao = int(numpy.sqrt(self.shape[1]))
        v = numpy.asarray(self.dat[p0:p1])
        v = lib.transpose(v.reshape(-1,nao,nao), axes=(0,2,1)).conj()
        return v.reshape(-1,nao**2)

