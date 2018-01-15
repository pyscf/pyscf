#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.

Note when diffuse functions are used in fitting basis, it is easy to cause
linear dependence (non-positive definite) issue under PBC.

Ref:
'''

import time
import copy
import tempfile
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df import addons
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc import tools
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import aft
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.aft import estimate_eta, get_nuc
from pyscf.pbc.df.df_jk import zdotCN, zdotNN, zdotNC
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member, unique

LINEAR_DEP_THR = 1e-9

def make_modrho_basis(cell, auxbasis=None, drop_eta=1.):
    auxcell = addons.make_auxmol(cell, auxbasis)

# Note libcint library will multiply the norm of the integration over spheric
# part sqrt(4pi) to the basis.
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    steep_shls = []
    ndrop = 0
    rcut = []
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

            r = _estimate_rcut(es, l, abs(cs).max(axis=1), cell.precision)
            rcut.append(r.max())

    auxcell.rcut = max(rcut)

    auxcell._bas = numpy.asarray(auxcell._bas[steep_shls], order='C')
    logger.info(cell, 'Drop %d primitive fitting functions', ndrop)
    logger.info(cell, 'make aux basis, num shells = %d, num cGTOs = %d',
                auxcell.nbas, auxcell.nao_nr())
    logger.info(cell, 'auxcell.rcut %s', auxcell.rcut)
    return auxcell

def make_modchg_basis(auxcell, smooth_eta):
# * chgcell defines smooth gaussian functions for each angular momentum for
#   auxcell. The smooth functions may be used to carry the charge
    chgcell = copy.copy(auxcell)  # smooth model density for coulomb integral to carry charge
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    chg_bas = []
    chg_env = [smooth_eta]
    ptr_eta = auxcell._env.size
    ptr = ptr_eta + 1
    l_max = auxcell._bas[:,gto.ANG_OF].max()
# _gaussian_int(l*2+2) for multipole integral:
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
    norms = [half_sph_norm/gto.mole._gaussian_int(l*2+2, smooth_eta)
             for l in range(l_max+1)]
    for ia in range(auxcell.natm):
        for l in set(auxcell._bas[auxcell._bas[:,gto.ATOM_OF]==ia, gto.ANG_OF]):
            chg_bas.append([ia, l, 1, 1, 0, ptr_eta, ptr, 0])
            chg_env.append(norms[l])
            ptr += 1

    chgcell._atm = auxcell._atm
    chgcell._bas = numpy.asarray(chg_bas, dtype=numpy.int32).reshape(-1,gto.BAS_SLOTS)
    chgcell._env = numpy.hstack((auxcell._env, chg_env))
    chgcell.rcut = _estimate_rcut(smooth_eta, l_max, 1., auxcell.precision)
    logger.debug1(auxcell, 'make compensating basis, num shells = %d, num cGTOs = %d',
                  chgcell.nbas, chgcell.nao_nr())
    logger.debug1(auxcell, 'chgcell.rcut %s', chgcell.rcut)
    return chgcell

# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)
    outcore.aux_e2(cell, fused_cell, cderi_file, 'int3c2e_sph', aosym='s2',
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
    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('int2c2e_sph', hermi=1, kpts=uniq_kpts)
    feri = h5py.File(cderi_file)

# An alternative method to evalute j2c. This method might have larger numerical error?
#    chgcell = make_modchg_basis(auxcell, mydf.eta)
#    for k, kpt in enumerate(uniq_kpts):
#        aoaux = ft_ao.ft_ao(chgcell, Gv, None, b, gxyz, Gvbase, kpt).T
#        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, gs))
#        LkR = aoaux.real * coulG
#        LkI = aoaux.imag * coulG
#        j2caux = numpy.zeros_like(j2c[k])
#        j2caux[naux:,naux:] = j2c[k][naux:,naux:]
#        if is_zero(kpt):  # kpti == kptj
#            j2caux[naux:,naux:] -= lib.ddot(LkR, LkR.T)
#            j2caux[naux:,naux:] -= lib.ddot(LkI, LkI.T)
#            j2c[k] = j2c[k][:naux,:naux] - fuse(fuse(j2caux.T).T)
#            vbar = fuse(mydf.auxbar(fused_cell))
#            s = (vbar != 0).astype(numpy.double)
#            j2c[k] -= numpy.einsum('i,j->ij', vbar, s)
#            j2c[k] -= numpy.einsum('i,j->ij', s, vbar)
#        else:
#            j2cR, j2cI = zdotCN(LkR, LkI, LkR.T, LkI.T)
#            j2caux[naux:,naux:] -= j2cR + j2cI * 1j
#            j2c[k] = j2c[k][:naux,:naux] - fuse(fuse(j2caux.T).T)
#        feri['j2c/%d'%k] = fuse(fuse(j2c[k]).T).T
#        aoaux = LkR = LkI = coulG = None

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
        feri['j2c/%d'%k] = fuse(fuse(j2c[k]).T).T
        aoaux = LkR = LkI = coulG = None
    j2c = None

    def make_kpt(uniq_kptji_id):  # kpt = kptj - kpti
        kpt = uniq_kpts[uniq_kptji_id]
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        Gaux *= mydf.weighted_coulG(kpt, False, gs).reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        j2c = numpy.asarray(feri['j2c/%d'%uniq_kptji_id])
        try:
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError as e:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that gs is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([e.message, msg]))
            w, v = scipy.linalg.eigh(j2c)
            log.debug('DF metric linear dependency for kpt %s', uniq_kptji_id)
            log.debug('cond = %.4g, drop %d bfns',
                      w[-1]/w[0], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v = v[:,w>mydf.linear_dep_threshold].T.conj()
            v /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c = v
            j2ctag = 'eig'
        naux0 = j2c.shape[0]

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=adapted_kptjs)
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
        buf = numpy.empty(nkptj*buflen*Gblksize, dtype=numpy.complex128)

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
            v = None

            shls_slice = (bstart, bend, 0, bend)
            for p0, p1 in lib.prange(0, ngs, Gblksize):
                dat = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                            b, gxyz[p0:p1], Gvbase, kpt,
                                            adapted_kptjs, out=buf)
                nG = p1 - p0
                for k, ji in enumerate(adapted_ji_idx):
                    aoao = dat[k].reshape(nG,ncol)
                    pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.T
                    pqkI[:] = aoao.imag.T

                    lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k][naux:], 1)
                    lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k][naux:], 1)
                    if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                        lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k][naux:], 1)
                        lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k][naux:], 1)

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = fuse(j3cR[k])
                else:
                    v = fuse(j3cR[k] + j3cI[k] * 1j)
                if j2ctag == 'CD':
                    v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                else:
                    v = lib.dot(j2c, v)
                feri['j3c/%d'%ji][:naux0,col0:col1] = v

        del(feri['j2c/%d'%uniq_kptji_id])
        for k, ji in enumerate(adapted_ji_idx):
            v = feri['j3c/%d'%ji][:naux0]
            del(feri['j3c/%d'%ji])
            feri['j3c/%d'%ji] = v

    for k, kpt in enumerate(uniq_kpts):
        make_kpt(k)

    feri.close()


class GDF(aft.AFTDF):
    '''Gaussian density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.kpts_band = None
        self.auxbasis = None
        if cell.dimension == 0:
            self.eta = 0.2
            self.gs = cell.gs
        else:
            ke_cutoff = tools.gs_to_cutoff(cell.lattice_vectors(), cell.gs)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
            self.eta = min(aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision),
                           estimate_eta(cell, cell.precision))
            ke_cutoff = aft.estimate_ke_cutoff_for_eta(cell, self.eta, cell.precision)
            self.gs = tools.cutoff_to_gs(cell.lattice_vectors(), ke_cutoff)
            self.gs[cell.dimension:] = cell.gs[cell.dimension:]

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.blockdim = 240
        self.linear_dep_threshold = LINEAR_DEP_THR
        self._j_only = False
# If _cderi_to_save is specified, the 3C-integral tensor will be saved in this file.
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
# If _cderi is specified, the 3C-integral tensor will be read from this file
        self._cderi = None
        self._keys = set(self.__dict__.keys())

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        self._auxbasis = x
        self.auxcell = None
        self._cderi = None
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)

    def dump_flags(self, log=None):
        log = logger.new_logger(self, log)
        log.info('\n')
        log.info('******** %s flags ********', self.__class__)
        log.info('gs = %s', self.gs)
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
        log.info('eta = %s', self.eta)
        if isinstance(self._cderi, str):
            log.info('_cderi = %s  where DF integrals are loaded (readonly).',
                     self._cderi)
        elif isinstance(self._cderi_to_save, str):
            log.info('_cderi_to_save = %s', self._cderi_to_save)
        else:
            log.info('_cderi_to_save = %s', self._cderi_to_save.name)
        log.info('len(kpts) = %d', len(self.kpts))
        log.debug1('    kpts = %s', self.kpts)
        if self.kpts_band is not None:
            log.info('len(kpts_band) = %d', len(self.kpts_band))
            log.debug1('    kpts_band = %s', self.kpts_band)
        return self

    def check_sanity(self):
        return lib.StreamObject.check_sanity(self)

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

        self.auxcell = make_modrho_basis(self.cell, self.auxbasis, self.eta)

        if self.kpts_band is None:
            kpts = self.kpts
            kband_uniq = numpy.zeros((0,3))
        else:
            kpts = self.kpts
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
                if isinstance(self._cderi, str):
                    logger.warn(self, 'Value of _cderi is ignored. DF '
                                'integrals will be saved in file %s .', cderi)
            else:
                cderi = self._cderi_to_save.name
            self._cderi = cderi
            t1 = (time.clock(), time.time())
            self._make_j3c(self.cell, self.auxcell, kptij_lst, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    _make_j3c = _make_j3c

    def has_kpts(self, kpts):
        if kpts is None:
            return True
        else:
            kpts = numpy.asarray(kpts).reshape(-1,3)
            if self.kpts_band is None:
                return all((len(member(kpt, self.kpts))>0) for kpt in kpts)
            else:
                return all((len(member(kpt, self.kpts))>0 or
                            len(member(kpt, self.kpts_band))>0) for kpt in kpts)

    def auxbar(self, fused_cell=None):
        r'''
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
                compact=True, blksize=None):
        '''Short range part'''
        if self._cderi is None:
            self.build()
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        is_real = is_zero(kpti_kptj)
        nao = self.cell.nao_nr()
        if blksize is None:
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

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return df_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = df_ao2mo.get_eri
    ao2mo = get_mo_eri = df_ao2mo.general

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
        for LpqR, LpqI in self.sr_loop(compact=True, blksize=self.blockdim):
# LpqI should be 0 for gamma point DF
#            assert(numpy.linalg.norm(LpqI) < 1e-12)
            yield LpqR

    def get_naoaux(self):
# determine naoaux with self._cderi, because DF object may be used as CD
# object when self._cderi is provided.
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi, 'j3c/0') as feri:
            return feri.shape[0]

DF = GDF


def fuse_auxcell(mydf, auxcell):
    chgcell = make_modchg_basis(auxcell, mydf.eta)
    fused_cell = copy.copy(auxcell)
    fused_cell._atm, fused_cell._bas, fused_cell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)
    fused_cell.rcut = max(auxcell.rcut, chgcell.rcut)

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
        k_id = member(kpti_kptj, kptij_lst)
        if len(k_id) > 0:
            dat = self.feri['%s/%d' % (self.label,k_id[0])]
        else:
            # swap ki,kj due to the hermiticity
            kptji = kpti_kptj[[1,0]]
            k_id = member(kptji, kptij_lst)
            if len(k_id) == 0:
                raise RuntimeError('%s for kpts %s is not initialized.\n'
                                   'You need to update the attribute .kpts then call '
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
    def __getitem__(self, s):
        nao = int(numpy.sqrt(self.shape[1]))
        v = numpy.asarray(self.dat[s])
        v = lib.transpose(v.reshape(-1,nao,nao), axes=(0,2,1)).conj()
        return v.reshape(-1,nao**2)
