'''
Incore density fitting

Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import os
import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.agf2 import mpi_helper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.pbc.df import incore, ft_ao
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member, \
                                      unique, KPT_DIFF_TOL
from pyscf.pbc.df.df import make_modrho_basis, GDF, fuse_auxcell


def _get_kpt_hash(kpt, tol=KPT_DIFF_TOL):
    '''
    Get a hashable representation of the k-point up to a given tol to
    prevent the O(N_k) access cost.
    '''

    kpt_round = numpy.rint(numpy.asarray(kpt) / tol).astype(int)
    return tuple(kpt_round.ravel())


def _kconserve_indices(cell, uniq_kpts, kpt):
    '''
    Search which (kpts+kpt) satisfies momentum conservation.
    '''

    a = cell.lattice_vectors() / (2*numpy.pi)

    kdif = numpy.einsum('wx,ix->wi', a, uniq_kpts + kpt)
    kdif_int = numpy.rint(kdif)

    mask = numpy.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
    uniq_kptji_ids = numpy.where(mask)[0]

    return uniq_kptji_ids


def _get_2c2e(
        fused_cell,
        uniq_kpts,
        log):
    '''
    Get the bare two-center two-electron interaction, first term
    of Eq. 32.
    '''

    int2c2e = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    return int2c2e


def _get_3c2e(
        cell, fused_cell,
        kptij_lst,
        log):
    '''
    Get the bare three-center two-electron interaction, first term
    of Eq. 31.
    '''

    t1 = (time.clock(), time.time())

    nkij = len(kptij_lst)
    nao = cell.nao_nr()
    ngrids = fused_cell.nao_nr()
    aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)

    int3c2e = numpy.zeros((nkij, ngrids, nao*nao), dtype=numpy.complex128)

    for p0, p1 in mpi_helper.prange(0, fused_cell.nbas, fused_cell.nbas):
        log.debug2('3c2e part [%d -> %d] of %d' % (p0, p1, fused_cell.nbas))

        shls_slice = (0, cell.nbas, 0, cell.nbas, p0, p1)
        q0, q1 = aux_loc[p0], aux_loc[p1]

        int3c2e_part = incore.aux_e2(cell, fused_cell, 'int3c2e', aosym='s2',
                                     kptij_lst=kptij_lst, shls_slice=shls_slice)
        int3c2e_part = lib.transpose(int3c2e_part, axes=(0,2,1))

        if int3c2e_part.shape[-1] != nao*nao:
            assert int3c2e_part.shape[-1] == nao*(nao+1)//2
            int3c2e_part = lib.unpack_tril(int3c2e_part, lib.HERMITIAN, axis=-1)

        int3c2e_part = int3c2e_part.reshape((nkij, q1-q0, nao*nao))
        int3c2e[:,q0:q1] = int3c2e_part

        log.timer_debug1('3c2e part', *t1)

    mpi_helper.allreduce_safe_inplace(int3c2e)
    mpi_helper.barrier()

    return int3c2e


def _get_j2c(
        mydf,
        cell, auxcell, fused_cell, fuse,
        int2c2e,
        uniq_kpts,
        log):
    '''
    Build j2c using the 2c2e interaction, int2c2e, Eq. 32.
    '''

    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    b = cell.reciprocal_vectors()

    j2c = int2c2e.copy()

    for k, kpt in enumerate(uniq_kpts):
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        LkR = numpy.asarray(aoaux.real, order='C')
        LkI = numpy.asarray(aoaux.imag, order='C')
        aoaux = None

        # eq. 31 final three terms:
        if is_zero(kpt):  # kpti == kptj
            j2c[k][naux:] = int2c2e[k][naux:] - (
                                lib.ddot(LkR[naux:]*coulG, LkR.T) +
                                lib.ddot(LkI[naux:]*coulG, LkI.T))
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T
        else:
            j2cR, j2cI = zdotCN(LkR[naux:]*coulG,
                                LkI[naux:]*coulG, LkR.T, LkI.T)
            j2c[k][naux:] = int2c2e[k][naux:] - (j2cR + j2cI * 1j)
            j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T.conj()

        LkR = LkI = None
        coulG = None
        j2c[k] = fuse(fuse(j2c[k]).T).T

    return j2c


def _cholesky_decomposed_metric(
        mydf,
        cell,
        j2c,
        uniq_kptji_id,
        log):
    '''
    Get the Cholesky decomposed j2c.
    '''

    j2c_kpt = j2c[uniq_kptji_id]

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
        w = v = None
        j2ctag = 'eig'

    return j2c_kpt, j2ctag


def _get_j3c(
        mydf,
        cell, auxcell, fused_cell, fuse,
        j2c, int3c2e,
        uniq_kpts, uniq_inverse, kptij_lst,
        log,
        out=None):
    '''
    Build j2c using the 2c2e interaction, int2c2e, Eq. 31, and then
    contract with the Cholesky decomposed j2c.
    '''

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    b = cell.reciprocal_vectors()
    kptjs = kptij_lst[:,1]

    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2ctag = cholesky_j2c

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        Gaux *= mydf.weighted_coulG(kpt, False, mesh).reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        Gaux = None

        if is_zero(kpt): #and cell.dimension == 3:
            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=adapted_kptjs)
            ovlp = [numpy.ravel(s) for s in ovlp]

        shranges = balance_partition(cell.ao_loc_nr()*nao, nao*nao)
        pqkRbuf = numpy.empty(nao*nao*ngrids)
        pqkIbuf = numpy.empty(nao*nao*ngrids)
        buf = numpy.empty(nkptj*nao*nao*ngrids, dtype=numpy.complex128)

        bstart, bend, ncol = shranges[0]
        log.debug1('int3c2e')
        shls_slice = (bstart, bend, 0, cell.nbas)
        dat = ft_ao.ft_aopair_kpts(cell, Gv, shls_slice, 's1',
                                   b, gxyz, Gvbase, kpt,
                                   adapted_kptjs, out=buf)

        for k, ji in enumerate(adapted_ji_idx):
            v = int3c2e[ji]  #FIXME copy needed??

            # eq. 31 second term:
            if is_zero(kpt): # and cell.dimension == 3:
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

            # eq. 31 final term:
            lib.dot(kLR.T, pqkR.T, -1, j3cR[naux:], 1)
            lib.dot(kLI.T, pqkI.T, -1, j3cR[naux:], 1)
            if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                lib.dot(kLR.T, pqkI.T, -1, j3cI[naux:], 1)
                lib.dot(kLI.T, pqkR.T,  1, j3cI[naux:], 1)

            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                v = fuse(j3cR)
            else:
                v = fuse(j3cR + j3cI * 1j)

            if j2ctag == 'CD':
                v = scipy.linalg.solve_triangular(j2c, v, lower=True, 
                                                  overwrite_b=True)
                out[ji,:v.shape[0]] += v.reshape(-1, nao*nao)
            else:
                v = lib.dot(j2c, v)
                out[ji,:v.shape[0]] += v.reshape(-1, nao*nao)

        j3cR = j3cI = None

    if out is None:
        out = numpy.zeros((len(kptij_lst), naux, nao*nao), dtype=numpy.complex128)

    for k in mpi_helper.nrange(len(uniq_kpts)):
        kpt = uniq_kpts[k]
        # ensure k/-k symmetry in j2c dims:
        uniq_kptji_id = _kconserve_indices(cell, uniq_kpts, -kpt)[0]

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        cholesky_j2c = _cholesky_decomposed_metric(mydf, cell, j2c, 
                                                   uniq_kptji_id, log)

        log.debug1("make_kpt for kpt %s", k)
        make_kpt(k, cholesky_j2c)

    mpi_helper.allreduce_safe_inplace(out)
    mpi_helper.barrier()

    return out


def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    '''
    Build the j3c array.

    cell: the unit cell for the calculation
    auxcell: the unit cell for the auxiliary functions
    chgcell: the unit cell for the smooth Gaussians
    fused_cell: auxcell and chgcell combined
    '''

    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)

    if cell.dimension < 3:
        raise ValueError('IncoreGDF does not support low-dimension cells')

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

    # Get the 3c2e interaction:
    int3c2e = _get_3c2e(cell, fused_cell, kptij_lst, log)
    t1 = log.timer_debug1('_get_3c2e', *t1)

    # Get the 2c2e interaction:
    int2c2e = _get_2c2e(fused_cell, uniq_kpts, log)
    t1 = log.timer_debug1('_get_2c2e', *t1)

    # Get j2c:
    j2c = _get_j2c(mydf, cell, auxcell, fused_cell,
                   fuse, int2c2e, uniq_kpts, log)
    t1 = log.timer_debug1('_get_j2c', *t1)

    # Get j3c:
    j3c = _get_j3c(mydf, cell, auxcell, fused_cell, fuse, j2c, int3c2e,
                   uniq_kpts, uniq_inverse, kptij_lst, log)
    t1 = log.timer_debug1('_get_j3c', *t1)


    feri = {
        'j3c-kptij': kptij_lst,
        'j3c-kptij-hash': {},
        'j3c': j3c,
    }
    for k, kpt in enumerate(kptij_lst):
        val = _get_kpt_hash(kpt)
        feri['j3c-kptij-hash'][val] = feri['j3c-kptij-hash'].get(val, []) + [k,]

    return feri


class IncoreGDF(GDF):
    ''' Incore Gaussian density fitting
    '''

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
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) 
                                       for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = numpy.asarray(kptij_lst)

        if with_j3c:
            #TODO what to do with this? allow pickling?
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
        k_id = self._cderi['j3c-kptij-hash'].get(_get_kpt_hash(kpti_kptj), [])
        #if len(k_id):
        #    assert k_id[0] == member(kpti_kptj, self._cderi['j3c-kptij'])[0]

        if len(k_id) > 0:
            v = j3c[k_id[0]].copy()
        else:
            kptji = kpti_kptj[[1,0]]
            k_id = self._cderi['j3c-kptij-hash'].get(_get_kpt_hash(kptji), [])
            #assert k_id[0] == member(kptji, self._cderi['j3c-kptij'])[0]
            if len(k_id) == 0:
                raise RuntimeError('j3c for kpts %s is not initialized.\n'
                                   'You need to update the attribute .kpts '
                                   'then call .build().' % kpti_kptj)
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
