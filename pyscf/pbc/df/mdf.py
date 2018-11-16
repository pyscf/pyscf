#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Gaussian and planewaves mixed density fitting
Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import os
import time
import tempfile
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df import aft
from pyscf.pbc.df.df import fuse_auxcell
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf.pbc.df import mdf_jk
from pyscf.pbc.df import mdf_ao2mo
from pyscf import __config__


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell(mydf, auxcell)

    # Create swap file to avoid huge cderi_file. see also function
    # pyscf.pbc.df.df._make_j3c
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    outcore._aux_e2(cell, fused_cell, fswap, 'int3c2e', aosym='s2',
                    kptij_lst=kptij_lst, dataname='j3c-junk', max_memory=max_memory)
    t1 = log.timer_debug1('3c2e', *t1)

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
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        aoaux = fuse(aoaux)
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, mesh))
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
        fswap['j2c/%d'%k] = j2c_k
        aoaux = kLR = kLI = j2cR = j2cI = coulG = None
    j2c = None

    feri = h5py.File(cderi_file)
    feri['j3c-kptij'] = kptij_lst
    nsegs = len(fswap['j3c-junk/0'])
    def make_kpt(uniq_kptji_id):  # kpt = kptj - kpti
        kpt = uniq_kpts[uniq_kptji_id]
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        Gaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        Gaux = fuse(Gaux)
        Gaux *= mydf.weighted_coulG(kpt, False, mesh)
        kLR = Gaux.T.real.copy('C')
        kLI = Gaux.T.imag.copy('C')
        j2c = numpy.asarray(fswap['j2c/%d'%uniq_kptji_id])
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
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)
            for k, ji in enumerate(adapted_ji_idx):
                ovlp[k] = lib.pack_tril(ovlp[k])
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.2e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngrids, 16384)
        pqkRbuf = numpy.empty(buflen*Gblksize)
        pqkIbuf = numpy.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = numpy.empty((nkptj,buflen*Gblksize), dtype=numpy.complex128)
        def pw_contract(istep, sh_range, j3cR, j3cI):
            bstart, bend, ncol = sh_range
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            for p0, p1 in lib.prange(0, ngrids, Gblksize):
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
                feri['j3c/%d/%d'%(ji,istep)] = v

        with lib.call_in_background(pw_contract) as compute:
            col1 = 0
            for istep, sh_range in enumerate(shranges):
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d', \
                           istep+1, len(shranges), *sh_range)
                bstart, bend, ncol = sh_range
                col0, col1 = col1, col1+ncol
                j3cR = []
                j3cI = []
                for k, idx in enumerate(adapted_ji_idx):
                    v = [fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T for i in range(nsegs)]
                    v = fuse(numpy.vstack(v))
                    if is_zero(kpt) and cell.dimension == 3:
                        for i, c in enumerate(vbar):
                            if c != 0:
                                v[i] -= c * ovlp[k][col0:col1]
                    j3cR.append(numpy.asarray(v.real, order='C'))
                    if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                        j3cI.append(None)
                    else:
                        j3cI.append(numpy.asarray(v.imag, order='C'))
                    v = None
                compute(istep, sh_range, j3cR, j3cI)
        for ji in adapted_ji_idx:
            del(fswap['j3c-junk/%d'%ji])

    for k, kpt in enumerate(uniq_kpts):
        make_kpt(k)

    feri.close()


# valence_exp = 1. is the Gaussian typicall sits in valence
VALENCE_EXP = getattr(__config__, 'pbc_df_mdf_valence_exp', 1.0)
def _mesh_for_valence(cell, valence_exp=VALENCE_EXP):
    '''Energy cutoff estimation'''
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = numpy.linalg.norm(b[0]) / (2*numpy.pi)
    elif cell.dimension == 2:
        w = numpy.linalg.norm(numpy.cross(b[0], b[1])) / (2*numpy.pi)**2
    else:
        w = abs(numpy.linalg.det(b)) / (2*numpy.pi)**3

    precision = cell.precision * 10
    Ecut_max = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i).copy()
        es[es>valence_exp] = valence_exp
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, precision, w)
        Ecut_max = max(Ecut_max, ke_guess.max())
    mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), Ecut_max)
    mesh = numpy.min((mesh, cell.mesh), axis=0)
    mesh[cell.dimension:] = cell.mesh[cell.dimension:]
    return mesh
del(VALENCE_EXP)


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
        self._auxbasis = None
        self.mesh = _mesh_for_valence(cell)

        # In MDF, fitting PWs (self.mesh), and parameters eta and exp_to_discard
        # are related to each other. The compensated function does not need to
        # be very smooth. It just needs to be expanded by the specified PWs
        # (self.mesh). self.eta is estimated on the fly based on the value of
        # self.mesh.
        self.eta = None

        # Any functions which are more diffused than the compensated Gaussian
        # are linearly dependent to the PWs. They can be removed from the
        # auxiliary set without affecting the accuracy of MDF. exp_to_discard
        # can be set to the value of self.eta
        self.exp_to_discard = None

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.blockdim = getattr(__config__, 'df_df_DF_blockdim', 240)
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
            ke_cutoff = tools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
            return aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    @eta.setter
    def eta(self, x):
        self._eta = x

    @property
    def exp_to_discard(self):
        if self._exp_to_discard is not None:
            return self._exp_to_discard
        else:
            return self.eta
    @exp_to_discard.setter
    def exp_to_discard(self, x):
        self._exp_to_discard = x

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
    def loop(self, blksize=None):
        for dat in aft.AFTDF.loop(self, blksize):
            yield dat
        for dat in df.DF.loop(self, blksize):
            yield dat

    def get_naoaux(self):
        return df.DF.get_naoaux(self) + aft.AFTDF.get_naoaux(self)
