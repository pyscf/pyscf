#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

r'''
Range-separated Gaussian Density Fitting (RSGDF)

rsdf_builder.py uses a different algorithm to build RS-GDF tensors. Note both
modules CANNOT handle the long-range operator erf(omega*r12)/r12. It has to be
computed with the gdf_builder module.

ref.:
[1] For the RSGDF method:
    Hong-Zhou Ye and Timothy C. Berkelbach, J. Chem. Phys. 154, 131104 (2021).
[2] For the SR lattice sum integral screening:
    Hong-Zhou Ye and Timothy C. Berkelbach, arXiv:2107.09704.

In RSGDF, the two-center and three-center Coulomb integrals are calculated in two pars:
    j2c = j2c_SR(omega) + j2c_LR(omega)
    j3c = j3c_SR(omega) + j3c_LR(omega)
where the SR and LR integrals correspond to using the following potentials
    g_SR(r_12;omega) = erfc(omega * r_12) / r_12
    g_LR(r_12;omega) = erf(omega * r_12) / r_12
The SR integrals are evaluated in real space using a lattice summation, while
the LR integrals are evaluated in reciprocal space with a plane wave basis.
'''

import os
import h5py
import scipy.linalg
import tempfile
import numpy as np

from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.lib import parameters as param
from pyscf.pbc.df.df import GDF
from pyscf.pbc.df import aft, aft_jk
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import gdf_builder
from pyscf.pbc.df.incore import Int3cBuilder
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import (is_zero, member, unique,
                                       members_with_wrap_around)
from pyscf.df.addons import make_auxmol


def get_aux_chg(auxcell):
    r""" Compute charge of the auxiliary basis, \int_Omega dr chi_P(r)

    Returns:
        The function returns a 1d numpy array of size auxcell.nao_nr().
    """
    G0 = np.zeros((1, 3))
    return ft_ao.ft_ao(auxcell, G0)[0].real


class RSGDF(GDF):
    '''Range Separated Gaussian Density Fitting
    '''
    _keys = {
        'use_bvk', 'precision_R', 'precision_G', 'npw_max', '_omega_min',
        'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c', 'mesh_j2c',
        'precision_j2c', 'j2c_eig_always', 'kpts',
    }

    def weighted_coulG(self, kpt=np.zeros(3), exx=False, mesh=None, omega=None):
        return aft.weighted_coulG(self, kpt, exx, mesh, omega)

    def __init__(self, cell, kpts=np.zeros((1,3))):
        if cell.dimension < 3:
            raise NotImplementedError("""
RSGDF for low-dimensional systems are not available yet. We recommend using
cell.dimension=3 with large vacuum.""")

        # if True and kpts are gamma-inclusive, RSDF will use the bvk cell
        # trick for computing both j3c_SR and j3c_LR. If kpts are not
        # gamma-inclusive, this attribute will be ignored.
        self.use_bvk = True

        # precision for real-space lattice sum (R) and reciprocal-space
        # Fourier transform (G).
        self.precision_R = cell.precision * 1e-2
        self.precision_G = cell.precision

        # omega and PW mesh size for j3c.
        # 1. If 'omega' is given, the code can search an appropriate PW mesh of
        # size 'mesh_compact' that computes the LR-AFT of j3c to 'precision_G'.
        # 2. If 'omega' is not given, the code will search for the maximum
        # omega such that the size of 'mesh_compact' does not exceed 'npw_max'.
        # The default for 'npw_max' is 350 (i.e., 7x7x7 for a 3D cubic
        # lattice). If thus determined 'omega' is smaller than '_omega_min'
        # (default: 0.3), 'omega' will be set to '_omega_min' and 'mesh_compact'
        # is determined from the new 'omega' (ignoring 'npw_max').
        # Note 1: In both cases, the user can manually overwrite the
        # auto-determined 'mesh_compact'.
        # Note 2: 'ke_cutoff' is not an input option. Use 'mesh_compact' directly.
        self.npw_max = 350
        self._omega_min = 0.3
        self.omega = None
        self.ke_cutoff = None
        self.mesh_compact = None

        # omega and PW mesh size for j2c.
        # Like for j3c, if 'omega_j2c' is given, the code can determine an
        # appropriate PW mesh of size 'mesh_j2c' that computes the LR-AFT of j2c
        # to 'precision_j2c'.
        # The default ('omega_j2c' = 0.4 and 'precision_j2c' = 1e-14) is recommended.
        # Like for j3c, 'mesh_j2c' can be overwritten manually.
        self.omega_j2c = 0.4
        self.mesh_j2c = None
        self.precision_j2c = 1e-14

        # set True to force calculating j2c^(-1/2) using eigenvalue
        # decomposition (ED); otherwise, Cholesky decomposition (CD) is used
        # first, and ED is called only if CD fails.
        self.j2c_eig_always = False

        GDF.__init__(self, cell, kpts=kpts)

        self.kpts = np.reshape(self.kpts, (-1,3))

    def dump_flags(self, verbose=None):
        cell = self.cell
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('cell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 cell.nbas, cell.nao_nr(), cell.npgto_nr())
        log.info('use_bvk = %s', self.use_bvk)
        log.info('precision_R = %s', self.precision_R)
        log.info('precision_G = %s', self.precision_G)
        log.info('j2c_eig_always = %s', self.j2c_eig_always)
        log.info('omega = %s', self.omega)
        log.info('ke_cutoff = %s', self.ke_cutoff)
        if self.mesh is not None:
            log.info('mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        log.info('mesh_compact = %s (%d PWs)', self.mesh_compact,
                 np.prod(self.mesh_compact))
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
            log.info('auxcell precision= %s', self.auxcell.precision)
            log.info('auxcell rcut = %s', self.auxcell.rcut)
            log.info('omega_j2c = %s', self.omega_j2c)
            log.info('mesh_j2c = %s (%d PWs)', self.mesh_j2c,
                     np.prod(self.mesh_j2c))

        auxcell = self.auxcell
        log.info('auxcell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 auxcell.nbas, auxcell.nao_nr(),
                 auxcell.npgto_nr())

        log.info('exp_to_discard = %s', self.exp_to_discard)
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

    def _rs_build(self):
        log = logger.Logger(self.stdout, self.verbose)

        # find kmax
        kpts = self.kpts if self.kpts_band is None else np.vstack(
                                                    [self.kpts, self.kpts_band])
        b = self.cell.reciprocal_vectors()
        scaled_kpts = np.linalg.solve(b.T, kpts.T).T
        scaled_kpts[scaled_kpts > 0.49999999] -= 1
        kpts = np.dot(scaled_kpts, b)
        kmax = np.linalg.norm(kpts, axis=-1).max()
        scaled_kpts = kpts = None
        if kmax < 1.e-3: kmax = (0.75/np.pi/self.cell.vol)**0.33333333*2*np.pi

        # If omega is not given, estimate it from npw_max
        r2o = True
        if self.omega is None:
            self.omega, self.ke_cutoff, mesh_compact = \
                                rsdf_helper.estimate_omega_for_npw(
                                                self.cell, self.npw_max,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)
            # if omega from npw_max is too small, use omega_min
            if self.omega < self._omega_min:
                self.omega = self._omega_min
                self.ke_cutoff, mesh_compact = \
                                    rsdf_helper.estimate_mesh_for_omega(
                                                    self.cell, self.omega,
                                                    self.precision_G,
                                                    kmax=kmax,
                                                    round2odd=r2o)
            # Use the thus determined mesh_compact only if not p[rovided
            if self.mesh_compact is None:
                self.mesh_compact = mesh_compact
        # If omega is provided but mesh_compact is not
        elif self.mesh_compact is None:
            self.ke_cutoff, self.mesh_compact = \
                                rsdf_helper.estimate_mesh_for_omega(
                                                self.cell, self.omega,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)

        self.mesh_compact = self.cell.symmetrize_mesh(self.mesh_compact)

        # build auxcell
        auxcell = make_auxmol(self.cell, self.auxbasis)
        # drop exponents
        drop_eta = self.exp_to_discard
        if drop_eta is not None and drop_eta > 0:
            log.info("Drop primitive fitting functions with exponent < %s",
                     drop_eta)
            auxbasis = rsdf_helper.remove_exp_basis(auxcell._basis,
                                                    amin=drop_eta)
            auxcellnew = make_auxmol(self.cell, auxbasis)
            auxcell = auxcellnew

        # determine mesh for computing j2c
        auxcell.precision = self.precision_j2c
        auxcell.rcut = max([auxcell.bas_rcut(ib, auxcell.precision)
                            for ib in range(auxcell.nbas)])
        if self.mesh_j2c is None:
            self.mesh_j2c = rsdf_helper.estimate_mesh_for_omega(
                                    auxcell, self.omega_j2c, round2odd=True)[1]
        self.mesh_j2c = self.cell.symmetrize_mesh(self.mesh_j2c)
        self.auxcell = auxcell

    def _kpts_build(self, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(np.vstack((self.kpts_band,kpts_band)))[0]

    def _gdf_build(self, j_only=None, with_j3c=True):
        if j_only is not None:
            self._j_only = j_only

        if with_j3c:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'File %s (specified by ._cderi) is '
                                'overwritten by GDF initialization.', cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .', cderi)
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]
        dfbuilder = _RSGDFBuilder(cell, auxcell, kpts_union)
        dfbuilder.__dict__.update(self.__dict__)
        dfbuilder.kpts = kpts_union
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        # formatting k-points
        self._kpts_build(kpts_band=kpts_band)

        # build for range-separation hybrid
        self._rs_build()

        # dump flags before the final build
        self.check_sanity()
        self.dump_flags()

        # do normal gdf build with the modified _make_j3c
        self._gdf_build(j_only=j_only, with_j3c=with_j3c)

        return self


RSDF = RSGDF


class _RSGDFBuilder(rsdf_builder._RSGDFBuilder):
    _keys = {
        'use_bvk', 'precision_R', 'precision_G', 'npw_max', '_omega_min',
        'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c', 'mesh_j2c',
        'precision_j2c', 'j2c_eig_always', 'kpts',
    }

    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        self.eta = None
        self.mesh = None
        if cell.omega != 0:
            # Initialize omega to cell.omega for HF exchange of short range
            # int2e in RSH functionals
            self.omega = abs(cell.omega)
        else:
            self.omega = None
        self.ke_cutoff = None
        self.bvk_kmesh = None
        self.supmol_ft = None

        Int3cBuilder.__init__(self, cell, auxcell, kpts)

        # set True to force calculating j2c^(-1/2) using eigenvalue
        # decomposition (ED); otherwise, Cholesky decomposition (CD) is used
        # first, and ED is called only if CD fails.
        self.j2c_eig_always = False
        self.linear_dep_threshold = rsdf_builder.LINEAR_DEP_THR

    def build(self, omega=None):
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, rsdf_builder.RCUT_THRESHOLD, verbose=log)

        rcut = rsdf_builder.estimate_ft_rcut(rs_cell, cell.precision,
                                             exclude_dd_block=False)
        supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(rs_cell, kmesh,
                                                           rcut.max(), log)
        supmol_ft.exclude_dd_block = False
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                  supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        return self

    def get_2c2e(self, uniq_kpts):
        cell = self.cell
        auxcell = self.auxcell
        # compute j2c first as it informs the integral screening in computing j3c
        # short-range part of j2c ~ (-kpt_ji | kpt_ji)
        omega_j2c = abs(self.omega_j2c)
        j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=uniq_kpts)

        # get charge of auxbasis
        if cell.dimension == 3:
            qaux = get_aux_chg(auxcell)
        else:
            qaux = np.zeros(auxcell.nao_nr())

        # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
        qaux2 = None
        g0_j2c = np.pi/omega_j2c**2./cell.vol
        mesh_j2c = self.mesh_j2c
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        max_memory = max(2000, self.max_memory - lib.current_memory()[0])
        blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
        logger.debug2(self, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)

        for k, kpt in enumerate(uniq_kpts):
            # short-range charge part
            if is_zero(kpt) and cell.dimension == 3:
                if qaux2 is None:
                    qaux2 = np.outer(qaux,qaux)
                j2c[k] -= qaux2 * g0_j2c
            # long-range part via aft
            coulG_lr = self.weighted_coulG(kpt, mesh=mesh_j2c, omega=omega_j2c)
            for p0, p1 in lib.prange(0, ngrids, blksize):
                auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                auxGR = np.asarray(auxG.real, order='C')
                auxGI = np.asarray(auxG.imag, order='C')
                auxG = None

                if is_zero(kpt):  # kpti == kptj
                    j2c[k] += lib.ddot(auxGR*coulG_lr[p0:p1], auxGR.T)
                    j2c[k] += lib.ddot(auxGI*coulG_lr[p0:p1], auxGI.T)
                else:
                    j2cR, j2cI = zdotCN(auxGR*coulG_lr[p0:p1],
                                        auxGI*coulG_lr[p0:p1], auxGR.T, auxGI.T)
                    j2c[k] += j2cR + j2cI * 1j
                auxGR = auxGI = j2cR = j2cI = None
        return j2c

    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      kptij_lst=None, j_only=False, dataname='j3c-junk',
                      shls_slice=None):
        # Deadlock on NFS if you open an already-opened tmpfile in H5PY
        # swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
        fswap = lib.H5TmpFile(dir=os.path.dirname(cderi_file), prefix='.outcore_auxe2_swap')
        # avoid trash files
        os.unlink(fswap.filename)

        cell = self.cell
        if self.use_bvk and self.kpts_band is None:
            bvk_kmesh = self.bvk_kmesh
        else:
            bvk_kmesh = None

        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        rsdf_helper._aux_e2_nospltbas(cell, self.auxcell, self.omega,
                                      fswap, intor, aosym=aosym,
                                      kptij_lst=kptij_lst, dataname=dataname,
                                      max_memory=max_memory,
                                      bvk_kmesh=bvk_kmesh,
                                      precision=self.precision_R)
        return fswap

    def weighted_ft_ao(self, kpt):
        cell = self.cell
        auxcell = self.auxcell
        mesh = self.mesh_compact
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        shls_slice = (0, auxcell.nbas)
        auxG = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
        wcoulG_lr = self.weighted_coulG(kpt, mesh=mesh, omega=self.omega)
        auxG *= wcoulG_lr
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, ijlst_mapping, aosym):
        cell = self.cell
        kpts = self.kpts
        nkpts = len(self.kpts)
        vbar = None
        if is_zero(kpt) and cell.dimension == 3:
            qaux = get_aux_chg(self.auxcell)
            vbar = np.pi / self.omega**2 / cell.vol * qaux
            vbar_idx = np.where(vbar != 0)[0]
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
            if aosym == 's2':
                ovlp = [lib.pack_tril(s) for s in ovlp]
            else:
                ovlp = [s.ravel() for s in ovlp]

        # TODO: Store rs_density_fit cderi tensor in v1 format for the moment.
        # It should be changed to 'v2' format in the future.
        if ijlst_mapping is None:
            data_version = 'v2'
        else:
            data_version = 'v1'

        if data_version == 'v1':
            nsegs = len(h5group[f'j3c-junk/{ijlst_mapping[kpt_ij_idx[0]]}'])
        else:
            nsegs = len(h5group[f'j3c-junk/{kpt_ij_idx[0]}'])

        def load_j3c(col0, col1):
            j3cR = []
            j3cI = []
            for kk in kpt_ij_idx:
                if data_version == 'v1':
                    v = np.hstack([h5group[f'j3c-junk/{ijlst_mapping[kk]}/{i}'][0,col0:col1]
                                   for i in range(nsegs)])
                else:
                    v = np.hstack([h5group[f'j3c-junk/{kk}/{i}'][0,col0:col1]
                                   for i in range(nsegs)])
                vR = np.asarray(v.real, order='C')
                kj = kk % nkpts
                if is_zero(kpt) and is_zero(kpts[kj]):
                    vI = None
                else:
                    vI = np.asarray(v.imag, order='C')
                # vbar is the interaction between the background charge
                # and the auxiliary basis.  0D, 1D, 2D do not have vbar.
                if vbar is not None:
                    vmod = ovlp[kj][col0:col1,None] * vbar[vbar_idx]
                    vR[:,vbar_idx] -= vmod.real
                    if vI is not None:
                        vI[:,vbar_idx] -= vmod.imag
                j3cR.append(vR)
                j3cI.append(vI)
            return j3cR, j3cI

        return load_j3c

    def make_j3c(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                 j_only=False, dataname='j3c', shls_slice=None, kptij_lst=None):
        if self.cell.omega != 0:
            raise RuntimeError('RSGDF cannot be used to evaluate the long-range '
                               'HF exchange in RSH functionals.')

        cpu1 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.rs_cell is None:
            self.build()

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao
        naux = self.auxcell.nao

        if shls_slice is not None:
            raise NotImplementedError
        ish0, ish1 = 0, cell.nbas

        # ijlst_mapping maps the [nkpts x nkpts] kpts-pair to kpts-pair in
        # kptij_lst. Value -1 in ijlst_mapping means the kpts-pair does not
        # exist in kptij_lst
        ijlst_mapping = np.empty(nkpts * nkpts, dtype=int)
        ijlst_mapping[:] = -1
        if kptij_lst is None:
            if j_only:
                kpti_idx = np.arange(nkpts)
                ijlst_mapping[kpti_idx * nkpts + kpti_idx] = kpti_idx
                kptij_lst = np.concatenate([kpts[:,None,:], kpts[:,None,:]], axis=1)
                kk_idx = kpti_idx * nkpts + kpti_idx
            else:
                kpti_idx, kptj_idx = np.tril_indices(nkpts)
                nkpts_pair = kpti_idx.size
                ijlst_mapping[kpti_idx * nkpts + kptj_idx] = np.arange(nkpts_pair)
                kptij_lst = np.concatenate([kpts[kpti_idx,None,:],
                                            kpts[kptj_idx,None,:]], axis=1)
                kk_idx = kpti_idx * nkpts + kptj_idx
        else:
            kpti_idx = members_with_wrap_around(cell, kptij_lst[:,0], kpts)
            kptj_idx = members_with_wrap_around(cell, kptij_lst[:,1], kpts)
            ijlst_mapping[kpti_idx * nkpts + kptj_idx] = np.arange(len(kptij_lst))
            kk_idx = kpti_idx * nkpts + kptj_idx

        # TODO: Store rs_density_fit cderi tensor in v1 format for the moment.
        # It should be changed to 'v2' format in the future.
        data_version = 'v1'
        if h5py.is_hdf5(cderi_file):
            feri = lib.H5FileWrap(cderi_file, 'a')
            if 'kpts' in feri:
                del feri['j3c-kptij']
            if dataname in feri:
                log.warn(f'Overwritting {dataname} in {cderi_file}.')
                del feri[dataname]
        else:
            feri = lib.H5FileWrap(cderi_file, 'w')
        feri['j3c-kptij'] = kptij_lst

        fswap = self.outcore_auxe2(cderi_file, intor, aosym, comp,
                                   kptij_lst, j_only, 'j3c-junk', shls_slice)
        cpu1 = log.timer_debug1('3c2e', *cpu1)

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                               verbose=log)

        # recompute g0 and Gvectors for j3c
        mesh = self.mesh_compact
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
        tspans = np.zeros((3,2))    # lr, j2c_inv, j2c_cntr
        tspannames = ["ftaop+pw", "j2c_inv", "j2c_cntr"]
        def make_cderi(kpt, kpt_ij_idx, j2c):
            log.debug1('make_cderi for %s', kpt)
            kptjs = kpts[kpt_ij_idx % nkpts]
            nkptj = len(kptjs)
            if data_version == 'v1':
                input_kptij_idx = ijlst_mapping[kpt_ij_idx]
                # filter kpt_ij_idx, keeps only the kpts-pair in kptij_lst
                kpt_ij_idx = kpt_ij_idx[input_kptij_idx >= 0]
                # input_kptij_idx saves the indices of remaining kpts-pair in kptij_lst
                input_kptij_idx = input_kptij_idx[input_kptij_idx >= 0]
                log.debug1('kpt_ij_idx = %s', kpt_ij_idx)
                log.debug1('input_kptij_idx = %s', input_kptij_idx)
            else:
                input_kptij_idx = kpt_ij_idx
            if kpt_ij_idx.size == 0:
                return

            Gaux = self.weighted_ft_ao(kpt)

            if is_zero(kpt):  # kpti == kptj
                aosym = 's2'
                nao_pair = nao*(nao+1)//2
            else:
                aosym = 's1'
                nao_pair = nao**2

            load = self.gen_j3c_loader(fswap, kpt, kpt_ij_idx, ijlst_mapping, aosym)

            mem_now = lib.current_memory()[0]
            log.debug2('memory = %s', mem_now)
            max_memory = max(1000, self.max_memory - mem_now)
            # nkptj for 3c-coulomb arrays plus 1 Lpq array
            buflen = min(max(int(max_memory*.3e6/16/naux/(nkptj+1)), 1), nao_pair)
            sh_ranges = _guess_shell_ranges(cell, buflen, aosym, start=ish0, stop=ish1)
            buflen = max([x[2] for x in sh_ranges])
            # * 2 for the buffer used in preload
            max_memory -= buflen * naux * (nkptj+1) * 16e-6 * 2

            # +1 for a pqkbuf
            Gblksize = max(16, int(max_memory*1e6/16/buflen/(nkptj+1)))
            Gblksize = min(Gblksize, ngrids, 200000)

            cols = [sh_range[2] for sh_range in sh_ranges]
            locs = np.append(0, np.cumsum(cols))
            # buf for ft_aopair
            buf = np.empty(nkptj*buflen*Gblksize, dtype=np.complex128)
            for istep, j3c in enumerate(lib.map_with_prefetch(load, locs[:-1], locs[1:])):
                bstart, bend, ncol = sh_ranges[istep]
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep+1, len(sh_ranges), bstart, bend, ncol)
                if aosym == 's2':
                    shls_slice = (bstart, bend, 0, bend)
                else:
                    shls_slice = (bstart, bend, 0, cell.nbas)

                if self.has_long_range():
                    tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
                    for p0, p1 in lib.prange(0, ngrids, Gblksize):
                        # shape of Gpq (nkpts, nGv, ni, nj)
                        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt,
                                      kptjs, shls_slice, aosym, out=buf)
                        self.add_ft_j3c(j3c, Gpq, Gaux, p0, p1)
                        Gpq = None
                    tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[0] += tock_ - tick_

                j3cR, j3cI = j3c
                for k, idx in enumerate(input_kptij_idx):
                    cderi, cderi_negative = self.solve_cderi(j2c, j3cR[k], j3cI[k])
                    feri[f'{dataname}/{idx}/{istep}'] = cderi
                    if cderi_negative is not None:
                        # for low-dimension systems
                        feri[f'{dataname}-/{idx}/{istep}'] = cderi_negative
                j3cR = j3cI = j3c = cderi = None
                tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[2] += tick_ - tock_

        for kpt, kpt_ij_idx, cd_j2c \
                in self.gen_uniq_kpts_groups(j_only, fswap, kk_idx=kk_idx):
            make_cderi(kpt, kpt_ij_idx, cd_j2c)

        feri.close()
        # report time for aft part
        for tspan, tspanname in zip(tspans, tspannames):
            log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                       "%10s"%tspanname, *tspan)
        return self
