#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Gaussian and planewaves mixed density fitting
Ref:
'''

import time
import tempfile
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import tools
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df import aft
from pyscf.pbc.df.df import make_modrho_basis, fuse_auxcell
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf.pbc.df import mdf_jk
from pyscf.pbc.df import mdf_ao2mo


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

    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        aoaux = fuse(aoaux)
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, gs))
        kLR = (aoaux.real * coulG).T
        kLI = (aoaux.imag * coulG).T
        if not kLR.flags.c_contiguous: kLR = lib.transpose(kLR.T)
        if not kLI.flags.c_contiguous: kLI = lib.transpose(kLI.T)

        j2c_k = fuse(fuse(j2c[k]).T).T.copy()
        if is_zero(kpt):  # kpti == kptj
            j2c_k -= lib.dot(kLR.T, kLR)
            j2c_k -= lib.dot(kLI.T, kLI)
        else:
             # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
            j2cR, j2cI = zdotCN(kLR.T, kLI.T, kLR, kLI)
            j2c_k -= j2cR + j2cI * 1j
        feri['j2c/%d'%k] = j2c_k
        aoaux = kLR = kLI = j2cR = j2cI = coulG = None
    j2c = None

    def make_kpt(uniq_kptji_id):  # kpt = kptj - kpti
        kpt = uniq_kpts[uniq_kptji_id]
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        Gaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        Gaux = fuse(Gaux)
        Gaux *= mydf.weighted_coulG(kpt, False, gs)
        kLR = Gaux.T.real.copy('C')
        kLI = Gaux.T.imag.copy('C')
        j2c = numpy.asarray(feri['j2c/%d'%uniq_kptji_id])
# Note large difference may be found in results between the CD/eig treatments.
# In some systems, small integral errors can lead to different treatments of
# linear dependency which can be observed in the total energy/orbital energy
# around 4th decimal place.
#        try:
#            j2c = scipy.linalg.cholesky(j2c, lower=True)
#            j2ctag = 'CD'
#        except scipy.linalg.LinAlgError as e:
#
# Abandon CD treatment for better numerical stablity
        w, v = scipy.linalg.eigh(j2c)
        log.debug('MDF metric for kpt %s cond = %.4g, drop %d bfns',
                  uniq_kptji_id, w[-1]/w[0], numpy.count_nonzero(w<mydf.linear_dep_threshold))
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
        buf = numpy.empty((nkptj,buflen*Gblksize), dtype=numpy.complex128)

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

                    lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k], 1)
                    lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k], 1)
                    if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                        lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k], 1)
                        lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k], 1)

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = j3cR[k]
                else:
                    v = j3cR[k] + j3cI[k] * 1j
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


class MDF(df.DF):
    '''Gaussian and planewaves mixed density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.kpts_band = None
        self.auxbasis = None
        self.gs = cell.gs
        self.eta = None

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.blockdim = 240
        self.linear_dep_threshold = df.LINEAR_DEP_THR
        self._j_only = False
# If _cderi_to_save is specified, the 3C-integral tensor will be saved in this file.
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
# If _cderi is specified, the 3C-integral tensor will be read from this file
        self._cderi = None
        self._keys = set(self.__dict__.keys())

    @property
    def eta(self):
        if self._eta is not None:
            return self._eta
        else:
            cell = self.cell
            if cell.dimension == 0:
                return 0.2
            ke_cutoff = tools.gs_to_cutoff(cell.lattice_vectors(), self.gs)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
            return aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    @eta.setter
    def eta(self, x):
        self._eta = x

    _make_j3c = _make_j3c

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
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
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
        for dat in aft.AFTDF.loop(self):
            yield dat
        for dat in df.DF.loop(self):
            yield dat

    def get_naoaux(self):
        return df.DF.get_naoaux(self) + aft.AFTDF.get_naoaux(self)
