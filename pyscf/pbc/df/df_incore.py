'''
Incore density fitting
'''

import os
import time
import copy
import ctypes
import warnings
import tempfile
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df import addons
from pyscf.agf2 import mpi_helper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc import tools
from pyscf.pbc.df import incore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import aft
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.aft import estimate_eta, get_nuc
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf.pbc.df.aft import _sub_df_jk_
from pyscf import __config__

LINEAR_DEP_THR = getattr(__config__, 'pbc_df_df_DF_lindep', 1e-9)
LONGRANGE_AFT_TURNOVER_THRESHOLD = 2.5
KPT_DIFF_TOL = getattr(__config__, 'pbc_lib_kpts_helper_kpt_diff_tol', 1e-6)

from pyscf.pbc.df.df import make_modrho_basis, make_modchg_basis, GDF, fuse_auxcell


def get_kpt_hash(kpt, tol=KPT_DIFF_TOL):
    ''' 
    Get a hashable representation of the k-point up to a given tol to
    prevent the O(N_k) access cost.
    '''
    kpt_round = numpy.rint(numpy.asarray(kpt) / tol)
    return hash(tuple(kpt_round.ravel()))


def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

    j3c = numpy.zeros((len(kptij_lst), fused_cell.nao_nr(), nao*nao),
                      dtype=numpy.complex128)
    for p0, p1 in mpi_helper.prange(0, fused_cell.nbas, fused_cell.nbas):
        shls_slice = (0, cell.nbas, 0, cell.nbas, p0, p1)
        aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)[:shls_slice[5]+1]
        q0, q1 = aux_loc[shls_slice[4]], aux_loc[shls_slice[5]]

        #NOTE dominant call:
        j3c_part = incore.aux_e2(cell, fused_cell, 'int3c2e', aosym='s2',
                                 kptij_lst=kptij_lst, shls_slice=shls_slice)
        j3c_part = lib.transpose(j3c_part, axes=(0,2,1))

        if j3c_part.shape[-1] != nao*nao:
            assert j3c_part.shape[-1] == nao*(nao+1)//2
            j3c_part = lib.unpack_tril(j3c_part, lib.HERMITIAN, axis=-1)

        j3c_part = j3c_part.reshape((len(kptij_lst), q1-q0, nao*nao))
        j3c[:,q0:q1] = j3c_part

    mpi_helper.allreduce_safe_inplace(j3c)
    mpi_helper.barrier()

    t1 = log.timer_debug1('3c2e', *t1)

    j2c = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    for k, kpt in enumerate(uniq_kpts):
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        LkR = numpy.asarray(aoaux.real, order='C')
        LkI = numpy.asarray(aoaux.imag, order='C')
        aoaux = None

        if is_zero(kpt):  # kpti == kptj
            j2c[k][naux:] -= lib.ddot(LkR[naux:]*coulG, LkR.T)
            j2c[k][naux:] -= lib.ddot(LkI[naux:]*coulG, LkI.T)
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T
        else:
            j2cR, j2cI = zdotCN(LkR[naux:]*coulG,
                                LkI[naux:]*coulG, LkR.T, LkI.T)
            j2c[k][naux:] -= j2cR + j2cI * 1j
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T.conj()

        LkR = LkI = None
        coulG = None
        j2c[k] = fuse(fuse(j2c[k]).T).T

    def cholesky_decomposed_metric(uniq_kptji_id):
        j2c_kpt = j2c[uniq_kptji_id]
        j2c_negative = None
        try:
            j2c_kpt = scipy.linalg.cholesky(j2c_kpt, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            w, v = scipy.linalg.eigh(j2c_kpt)
            log.debug('DF metric linear dependency for kpt %s', uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns',
                      w[-1]/w[0], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = v[:,w>mydf.linear_dep_threshold].conj().T
            v1 /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c_kpt = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = numpy.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]/numpy.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c_kpt, j2c_negative, j2ctag

    feri = {}
    feri['j3c-kptij'] = kptij_lst
    feri['j3c-kptij-hash'] = {}
    for k, kpt in enumerate(kptij_lst):
        val = get_kpt_hash(kpt)
        feri['j3c-kptij-hash'][val] = feri['j3c-kptij-hash'].get(val, []) + [k,]
    feri['j3c'] = numpy.zeros((len(kptij_lst), naux, nao*nao), dtype=complex) 

    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2c_negative, j2ctag = cholesky_j2c

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG = mydf.weighted_coulG(kpt, False, mesh)
        Gaux *= wcoulG.reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        Gaux = None

        aosym = 's1'
        nao_pair = nao**2
        if is_zero(kpt) and cell.dimension == 3:
            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=adapted_kptjs)
            ovlp = [numpy.ravel(s) for s in ovlp]

        shranges = balance_partition(cell.ao_loc_nr()*nao, nao_pair)
        pqkRbuf = numpy.empty(nao_pair*ngrids)
        pqkIbuf = numpy.empty(nao_pair*ngrids)
        buf = numpy.empty(nkptj*nao_pair*ngrids, dtype=numpy.complex128)

        bstart, bend, ncol = shranges[0]
        log.debug1('int3c2e')
        shls_slice = (bstart, bend, 0, cell.nbas)
        #NOTE also expensive:
        dat = ft_ao.ft_aopair_kpts(cell, Gv, shls_slice, aosym,
                                   b, gxyz, Gvbase, kpt,
                                   adapted_kptjs, out=buf)
        for k, ji in enumerate(adapted_ji_idx):
            v = j3c[ji]
            if is_zero(kpt) and cell.dimension == 3:
                for i in numpy.where(vbar != 0)[0]:
                    v[i] -= vbar[i] * ovlp[k]
            j3cR = numpy.asarray(v.real, order='C')
            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                j3cI = None
            else:
                j3cI = numpy.asarray(v.imag, order='C')
            v = None

            aoao = dat[k].reshape(ngrids,ncol)
            pqkR = numpy.ndarray((ncol,ngrids), buffer=pqkRbuf)
            pqkI = numpy.ndarray((ncol,ngrids), buffer=pqkIbuf)
            pqkR[:] = aoao.real.T
            pqkI[:] = aoao.imag.T

            lib.dot(kLR.T, pqkR.T, -1, j3cR[naux:], 1)
            lib.dot(kLI.T, pqkI.T, -1, j3cR[naux:], 1)
            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                lib.dot(kLR.T, pqkI.T, -1, j3cI[naux:].real, 1)
                lib.dot(kLI.T, pqkR.T,  1, j3cI[naux:].real, 1)

            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                v = fuse(j3cR.real)
            else:
                v = fuse(j3cR + j3cI * 1j)
            if j2ctag == 'CD':
                v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                feri['j3c'][ji,:v.shape[0]] += v.reshape(-1, nao_pair)
            else:
                v = lib.dot(j2c, v)
                feri['j3c'][ji,:v.shape[0]] += v.reshape(-1, nao_pair)

            # low-dimension systems
            if j2c_negative is not None:
                raise NotImplementedError('incore gdf low dimension')
        j3cR = j3cI = None

    def conj_j2c(cholesky_j2c):
        j2c, j2c_negative, j2ctag = cholesky_j2c
        if j2c_negative is None:
            return j2c.conj(), None, j2ctag
        else:
            return j2c.conj(), j2c_negative.conj(), j2ctag

    a = cell.lattice_vectors() / (2*numpy.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = numpy.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = numpy.rint(kdif)
        mask = numpy.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = numpy.where(mask)[0]
        return uniq_kptji_ids

    for k in mpi_helper.nrange(len(uniq_kpts)):
        kpt = uniq_kpts[k]
        uniq_kptji_id = kconserve_indices(-kpt)[0] # ensure k/-k symmetry in j2c dims

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        cholesky_j2c = cholesky_decomposed_metric(uniq_kptji_id)

        log.debug1("make_kpt for kpt %s", k)
        make_kpt(k, cholesky_j2c)

    mpi_helper.allreduce_safe_inplace(feri['j3c'])
    mpi_helper.barrier()

    return feri


class IncoreGDF(GDF):
    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = numpy.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(numpy.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = make_modrho_basis(self.cell, self.auxbasis,
                                         self.exp_to_discard)

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        uniq_idx = unique(self.kpts)[1]
        kpts = numpy.asarray(self.kpts)[uniq_idx]
        if self.kpts_band is None:
            kband_uniq = numpy.zeros((0,3))
        else:
            kband_uniq = [k for k in self.kpts_band if len(member(k, kpts))==0]
        if j_only is None:
            j_only = self._j_only
        if j_only:
            kall = numpy.vstack([kpts,kband_uniq])
            kptij_lst = numpy.hstack((kall,kall)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = numpy.asarray(kptij_lst)

        if with_j3c:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'DF integrals in %s (specified by '
                                '._cderi) is overwritten by GDF '
                                'initialization. ', cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .',
                                cderi)
            t1 = (time.clock(), time.time())
            self._cderi = self._make_j3c(self.cell, self.auxcell, kptij_lst, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    _make_j3c = _make_j3c
    
    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None):
        '''Short range part'''
        if self._cderi is None:
            self.build()
        kpti, kptj = kpti_kptj
        cell = self.cell
        is_real = is_zero(kpti_kptj)
        pack = is_zero(kpti-kptj) and compact
        nao = cell.nao_nr()
        if blksize is None:
            blksize = self.get_naoaux() # return as one block

        j3c = self._cderi['j3c']
        kpti_kptj = numpy.asarray(kpti_kptj)
        kptij = numpy.asarray(self._cderi['j3c-kptij'])
        #k_id = member(kpti_kptj, kptij) # O(nk), as in pyscf's original algo
        k_id = self._cderi['j3c-kptij-hash'].get(get_kpt_hash(kpti_kptj), [])

        if len(k_id) > 0:
            v = j3c[k_id[0]].copy()
        else:
            kptji = kpti_kptj[[1,0]]
            #k_id = member(kptji, kptij) # O(nk), as in pyscf's original algo
            k_id = self._cderi['j3c-kptij-hash'].get(get_kpt_hash(kptji), [])
            v = j3c[k_id[0]]

            shape = v.shape
            v = lib.transpose(v.reshape(-1,nao,nao), axes=(0,2,1)).conj()
            v = v.reshape(shape)

        v_r = numpy.array(v.real, order='C')
        if is_real:
            if pack:
                v_r = lib.pack_tril(v_r.reshape(-1, nao, nao), axis=-1)
            v_i = numpy.zeros_like(v_r)
        else:
            v_i = numpy.array(v.imag, order='C')
            if pack:
                v_r = lib.pack_tril(v_r.reshape(-1, nao, nao), axis=-1)
                v_i = lib.pack_tril(v_i.reshape(-1, nao, nao), axis=-1)
        v = None

        for p0, p1 in lib.prange(0, self.get_naoaux(), blksize):
            yield v_r[p0:p1], v_i[p0:p1], 1
        v_r = v_i = None

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise NotImplementedError('incore gdf low dim')


    def get_naoaux(self):
        if self._cderi is None:
            self.build()
        return self._cderi['j3c'].shape[1]

DF = GDF
IncoreDF = IncoreGDF
