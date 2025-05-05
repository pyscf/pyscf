#!/usr/bin/env python
# Copyright 2014-2019,2021 The PySCF Developers. All Rights Reserved.
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
Density fitting

Divide the 3-center Coulomb integrals to two parts.  Compute the local
part in real space, long range part in reciprocal space.

Note when diffuse functions are used in fitting basis, it is easy to cause
linear dependence (non-positive definite) issue under PBC.

Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import os
import ctypes
import warnings
import tempfile
import contextlib
import itertools
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
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import aft
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.aft import estimate_eta, _check_kpts
from pyscf.pbc.df.df_jk import zdotCN
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf.pbc.df.gdf_builder import libpbc, _CCGDFBuilder, _CCNucBuilder
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder, _RSNucBuilder, LINEAR_DEP_THR
from pyscf import __config__


def make_modrho_basis(cell, auxbasis=None, drop_eta=None):
    r'''Generate a cell object using the density fitting auxbasis as
    the basis set. The normalization coefficients of the auxiliary cell are
    different to the regular (square-norm) convention. To simplify the
    compensated charge algorithm, they are normalized against
    \int (r^l e^{-ar^2} r^2 dr
    '''
    auxcell = incore.make_auxcell(cell, auxbasis)

# Note libcint library will multiply the norm of the integration over spheric
# part sqrt(4pi) to the basis.
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    steep_shls = []
    ndrop = 0
    rcut = []
    _env = auxcell._env.copy()
    for ib in range(len(auxcell._bas)):
        l = auxcell.bas_angular(ib)
        np = auxcell.bas_nprim(ib)
        nc = auxcell.bas_nctr(ib)
        es = auxcell.bas_exp(ib)
        ptr = auxcell._bas[ib,gto.PTR_COEFF]
        cs = auxcell._env[ptr:ptr+np*nc].reshape(nc,np).T

        if drop_eta is not None and numpy.any(es < drop_eta):
            cs = cs[es>=drop_eta]
            es = es[es>=drop_eta]
            np, ndrop = len(es), ndrop+np-len(es)

        if np > 0:
            pe = auxcell._bas[ib,gto.PTR_EXP]
            auxcell._bas[ib,gto.NPRIM_OF] = np
            _env[pe:pe+np] = es
# int1 is the multipole value. l*2+2 is due to the radial part integral
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
            int1 = gto.gaussian_int(l*2+2, es)
            s = numpy.einsum('pi,p->i', cs, int1)
# The auxiliary basis normalization factor is not a must for density expansion.
# half_sph_norm here to normalize the monopole (charge).  This convention can
# simplify the formulism of \int \bar{\rho}, see function auxbar.
            cs = numpy.einsum('pi,i->pi', cs, half_sph_norm/s)
            _env[ptr:ptr+np*nc] = cs.T.ravel()

            steep_shls.append(ib)

            r = _estimate_rcut(es, l, abs(cs).max(axis=1), cell.precision)
            rcut.append(r.max())

    auxcell._env = _env
    auxcell.rcut = max(rcut)

    auxcell._bas = numpy.asarray(auxcell._bas[steep_shls], order='C')
    logger.info(cell, 'Drop %d primitive fitting functions', ndrop)
    logger.info(cell, 'make aux basis, num shells = %d, num cGTOs = %d',
                auxcell.nbas, auxcell.nao_nr())
    logger.info(cell, 'auxcell.rcut %s', auxcell.rcut)
    return auxcell

make_auxcell = make_modrho_basis


class GDF(lib.StreamObject, aft.AFTDFMixin):
    '''Gaussian density fitting
    '''
    blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)
    _dataname = 'j3c'
    # Call _CCGDFBuilder if applicable. _CCGDFBuilder is slower than
    # _RSGDFBuilder but numerically more close to previous versions
    _prefer_ccdf = False
    # If True, force using density matrix-based K-build
    force_dm_kbuild = False

    _keys = {
        'blockdim', 'force_dm_kbuild', 'cell', 'kpts', 'kpts_band', 'eta',
        'mesh', 'exp_to_discard', 'exxdiv', 'auxcell', 'linear_dep_threshold',
    }

    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        if isinstance(kpts, KPoints):
            kpts = kpts.kpts
        self.kpts = kpts  # default is gamma point
        self.kpts_band = None
        self._auxbasis = None

        self.eta = None
        self.mesh = None

        # exp_to_discard to remove diffused fitting functions. The diffused
        # fitting functions may cause linear dependency in DF metric. Removing
        # the fitting functions whose exponents are smaller than exp_to_discard
        # can improve the linear dependency issue. However, this parameter
        # affects the quality of the auxiliary basis. The default value of
        # this parameter was set to 0.2 in v1.5.1 or older and was changed to
        # 0 since v1.5.2.
        self.exp_to_discard = cell.exp_to_discard

        # The following attributes are not input options.
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.linear_dep_threshold = LINEAR_DEP_THR
        self._j_only = False
# If _cderi_to_save is specified, the 3C-integral tensor will be saved in this file.
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
# If _cderi is specified, the 3C-integral tensor will be read from this file
        self._cderi = None
        self._rsh_df = {}  # Range separated Coulomb DF objects

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('_cderi_to_save', '_cderi', '_rsh_df'), reset_state=True)

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self._auxbasis = x
            self.auxcell = None
            self._cderi = None

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.auxcell = None
        self._cderi = None
        self._rsh_df = {}
        return self

    @property
    def gs(self):
        return [n//2 for n in self.mesh]
    @gs.setter
    def gs(self, x):
        warnings.warn('Attribute .gs is deprecated. It is replaced by attribute .mesh.\n'
                      'mesh = the number of PWs (=2*gs+1) for each direction.')
        self.mesh = [2*n+1 for n in x]

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
        if self.eta is not None:
            log.info('eta = %s', self.eta)
        if self.mesh is not None:
            log.info('mesh = %s (%d PWs)', self.mesh, numpy.prod(self.mesh))
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

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if j_only is not None:
            self._j_only = j_only
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

        if with_j3c and self._cderi_to_save is not None:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'File %s (specified by ._cderi) is '
                                'overwritten by GDF initialization.', cderi)
                    os.remove(cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .', cderi)
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(numpy.vstack([self.kpts, self.kpts_band]))[0]

        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            dfbuilder = _CCGDFBuilder(cell, auxcell, kpts_union)
            dfbuilder.eta = self.eta
        else:
            dfbuilder = _RSGDFBuilder(cell, auxcell, kpts_union)
        dfbuilder.mesh = self.mesh
        dfbuilder.linear_dep_threshold = self.linear_dep_threshold
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

    def cderi_array(self, label=None):
        '''
        Returns CDERIArray object which provides numpy APIs to access cderi tensor.
        '''
        if label is None:
            label = self._dataname
        if self._cderi is None:
            self.build(j_only=self._j_only)
        return CDERIArray(self._cderi, label)

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

    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None, aux_slice=None):
        '''Short range part'''
        if self._cderi is None:
            self.build(j_only=self._j_only)
        cell = self.cell
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        is_real = is_zero(kpti_kptj)
        nao = cell.nao_nr()
        if blksize is None:
            if is_real:
                blksize = max_memory*1e6/8/(nao**2*2)
            else:
                blksize = max_memory*1e6/16/(nao**2*2)
            blksize /= 2  # For prefetch
            blksize = max(16, min(int(blksize), self.blockdim))
            logger.debug3(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

        def load(aux_slice):
            b0, b1 = aux_slice
            naux = b1 - b0
            if is_real:
                LpqR = numpy.asarray(j3c[b0:b1].real)
                if compact and LpqR.shape[1] == nao**2:
                    LpqR = lib.pack_tril(LpqR.reshape(naux,nao,nao))
                elif unpack and LpqR.shape[1] != nao**2:
                    LpqR = lib.unpack_tril(LpqR).reshape(naux,nao**2)
                LpqI = numpy.zeros_like(LpqR)
            else:
                Lpq = numpy.asarray(j3c[b0:b1])
                LpqR = numpy.asarray(Lpq.real, order='C')
                LpqI = numpy.asarray(Lpq.imag, order='C')
                Lpq = None
                if compact and LpqR.shape[1] == nao**2:
                    LpqR = lib.pack_tril(LpqR.reshape(naux,nao,nao))
                    LpqI = lib.pack_tril(LpqI.reshape(naux,nao,nao))
                elif unpack and LpqR.shape[1] != nao**2:
                    LpqR = lib.unpack_tril(LpqR).reshape(naux,nao**2)
                    LpqI = lib.unpack_tril(LpqI, lib.ANTIHERMI).reshape(naux,nao**2)
            return LpqR, LpqI

        with _load3c(self._cderi, self._dataname, kpti_kptj) as j3c:
            if aux_slice is None:
                slices = lib.prange(0, j3c.shape[0], blksize)
            else:
                slices = lib.prange(*aux_slice, blksize)
            for LpqR, LpqI in lib.map_with_prefetch(load, slices):
                yield LpqR, LpqI, 1
                LpqR = LpqI = None

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            # Truncated Coulomb operator is not positive definite. Load the
            # CDERI tensor of negative part.
            with _load3c(self._cderi, self._dataname+'-', kpti_kptj,
                         ignore_key_error=True) as j3c:
                if aux_slice is None:
                    slices = lib.prange(0, j3c.shape[0], blksize)
                else:
                    slices = lib.prange(*aux_slice, blksize)
                for LpqR, LpqI in lib.map_with_prefetch(load, slices):
                    yield LpqR, LpqI, -1
                    LpqR = LpqI = None

    def get_pp(self, kpts=None):
        '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
        '''
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            dfbuilder = _CCNucBuilder(cell, kpts).build()
        else:
            dfbuilder = _RSNucBuilder(cell, kpts).build()
        vpp = dfbuilder.get_pp()
        if is_single_kpt:
            vpp = vpp[0]
        return vpp

    def get_nuc(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCGDFBuilder is the only option
            dfbuilder = _CCNucBuilder(cell, kpts).build()
        else:
            dfbuilder = _RSNucBuilder(cell, kpts).build()
        nuc = dfbuilder.get_nuc()
        if is_single_kpt:
            nuc = nuc[0]
        return nuc

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None and omega != 0:  # J/K for RSH functionals
            cell = self.cell
            # * AFT is computationally more efficient than GDF if the Coulomb
            #   attenuation tends to the long-range role (i.e. small omega).
            # * Note: changing to AFT integrator may cause small difference to
            #   the GDF integrator.
            # * The sparse mesh is not appropriate for low dimensional systems
            #   with infinity vacuum since the ERI may require large mesh to
            #   sample density in vacuum.
            if (omega > 0 and
                cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum'):
                mydf = aft.AFTDF(cell, self.kpts)
                ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
            else:
                mydf = self
            with mydf.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            return df_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = df_ao2mo.get_eri
    ao2mo = get_mo_eri = df_ao2mo.general
    ao2mo_7d = df_ao2mo.ao2mo_7d

    def update_mp(self):
        mf = self.copy()
        mf.with_df = self
        return mf

    def update_cc(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def prange(self, start, stop, step):
        '''This is a hook for MPI parallelization. DO NOT use it out of the
        scope of AFTDF/GDF/MDF.
        '''
        return lib.prange(start, stop, step)

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        '''Creates a temporary density fitting object for RSH-DF integrals.
        In this context, only LR or SR integrals for mol and auxmol are computed.
        '''
        cell = self.cell
        if cell.dimension != 0:
            assert omega < 0

        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            rsh_df._dataname = f'{self._dataname}-sr/{key}'
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        auxcell = getattr(self, 'auxcell', None)

        cell_omega = cell.omega
        cell.omega = omega
        auxcell_omega = None
        if auxcell is not None:
            auxcell_omega = auxcell.omega
            auxcell.omega = omega

        assert rsh_df.cell.omega == omega
        if getattr(rsh_df, 'auxcell', None) is not None:
            assert rsh_df.auxcell.omega == omega

        try:
            yield rsh_df
        finally:
            cell.omega = cell_omega
            if auxcell_omega is not None:
                auxcell.omega = auxcell_omega

################################################################################
# With this function to mimic the molecular DF.loop function, the pbc gamma
# point DF object can be used in the molecular code
    def loop(self, blksize=None):
        cell = self.cell
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise RuntimeError('ERIs of PBC-2D systems are not positive '
                               'definite. Current API only supports positive '
                               'definite ERIs.')

        if blksize is None:
            blksize = self.blockdim
        for LpqR, LpqI, sign in self.sr_loop(compact=True, blksize=blksize):
            # LpqI should be 0 for gamma point DF
            # assert (numpy.linalg.norm(LpqI) < 1e-12)
            yield LpqR

    def get_naoaux(self):
        '''The dimension of auxiliary basis at gamma point'''
# determine naoaux with self._cderi, because DF object may be used as CD
# object when self._cderi is provided.
        if self._cderi is None:
            self.build(j_only=self._j_only)

        cell = self.cell
        if isinstance(self._cderi, numpy.ndarray):
            # self._cderi is likely offered by user. Ensure
            # cderi.shape = (nkpts,naux,nao_pair)
            nao = cell.nao
            if self._cderi.shape[-1] == nao:
                assert self._cderi.ndim == 4
                naux = self._cderi.shape[1]
            elif self._cderi.shape[-1] in (nao**2, nao*(nao+1)//2):
                assert self._cderi.ndim == 3
                naux = self._cderi.shape[1]
            else:
                raise RuntimeError('cderi shape')
            return naux

        # self._cderi['j3c/k_id/seg_id']
        with h5py.File(self._cderi, 'r') as feri:
            key = next(iter(feri[self._dataname].keys()))
            dat = feri[f'{self._dataname}/{key}']
            if isinstance(dat, h5py.Group):
                naux = dat['0'].shape[0]
            else:
                naux = dat.shape[0]

            if (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum' and
                f'{self._dataname}-' in feri):
                key = next(iter(feri[f'{self._dataname}-'].keys()))
                dat = feri[f'{self._dataname}-/{key}']
                if isinstance(dat, h5py.Group):
                    naux += dat['0'].shape[0]
                else:
                    naux += dat.shape[0]
        return naux

    to_gpu = lib.to_gpu

DF = GDF

class CDERIArray:
    '''
    Provide numpy APIs to access cderi tensor. This object can be viewed as an
    5-dimension array [kpt-i, kpt-j, aux-index, ao-i, ao-j]
    '''

    def __init__(self, data_group, label='j3c'):
        self._data_is_h5obj = isinstance(data_group, h5py.Group)
        if not self._data_is_h5obj:
            data_group = h5py.File(data_group, 'r')
        self.data_group = data_group
        if 'kpts' not in data_group:
            # TODO: Deprecate the v1 data format
            self._data_version = 'v1'
            self._cderi = data_group.file.filename
            self._label = label
            self._kptij_lst = data_group['j3c-kptij'][()]
            kpts = unique(self._kptij_lst[:,0])[0]
            self.nkpts = nkpts = len(kpts)
            if len(self._kptij_lst) not in (nkpts, nkpts**2, nkpts*(nkpts+1)//2):
                raise RuntimeError(f'Dimension error for CDERI {self._cderi}')
            return

        self._data_version = 'v2'
        aosym = data_group['aosym'][()]
        if isinstance(aosym, bytes):
            aosym = aosym.decode()
        self.aosym = aosym
        self.j3c = data_group[label]
        self.kpts = data_group['kpts'][:]
        self.nkpts = self.kpts.shape[0]
        self.naux = 0
        nao_pair = 0
        for dat in self.j3c.values():
            nao_pair = sum(x.shape[1] for x in dat.values())
            self.naux = dat['0'].shape[0]
            break
        if self.aosym == 's1':
            nao = int(nao_pair ** .5)
            assert nao ** 2 == nao_pair
            self.nao = nao
        elif self.aosym == 's2':
            self.nao = int((nao_pair * 2)**.5)
        else:
            raise NotImplementedError

    def __del__(self):
        if not self._data_is_h5obj:
            self.data_group.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self._data_is_h5obj:
            self.data_group.close()

    def __getitem__(self, slices):
        if isinstance(slices, tuple):
            ns = len(slices)
            if ns < 2:
                # patch (:) to slices
                slices = slices + (range(self.nkpts),) * (2 - ns)

            k_slices = slices[:2]
            a_slices = slices[2:]
            if isinstance(k_slices[0], int) and isinstance(k_slices[1], int):
                return self._load_one(k_slices[0], k_slices[1], ())
        else:
            k_slices = slices
            a_slices = ()
        out = numpy.empty((self.nkpts, self.nkpts), dtype=object)
        out[k_slices] = True

        for ki, kj in numpy.argwhere(out):
            out[ki,kj] = self._load_one(ki, kj, a_slices)
        return out[k_slices]

    def _load_one(self, ki, kj, slices):
        if self._data_version == 'v1':
            with _load3c(self._cderi, self._label) as fload:
                if len(self._kptij_lst) == self.nkpts:
                    # kptij_lst was generated with option j_only, leading to
                    # only the diagonal terms
                    kikj = ki
                    kpti, kptj = self._kptij_lst[kikj]
                elif len(self._kptij_lst) == self.nkpts**2:
                    kikj = ki * self.nkpts + kj
                    kpti, kptj = self._kptij_lst[kikj]
                elif ki >= kj:
                    kikj = ki*(ki+1)//2 + kj
                    kpti, kptj = self._kptij_lst[kikj]
                else:
                    kikj = kj*(kj+1)//2 + ki
                    kptj, kpti = self._kptij_lst[kikj]
                out = fload(kpti, kptj)
                return out[slices]

        kikj = ki * self.nkpts + kj
        kjki = kj * self.nkpts + ki
        if self.aosym == 's1' or kikj == kjki:
            dat = self.j3c[str(kikj)]
            nsegs = len(dat)
            out = _hstack_datasets([dat[str(i)] for i in range(nsegs)], slices)
        elif self.aosym == 's2':
            dat_ij = self.j3c[str(kikj)]
            dat_ji = self.j3c[str(kjki)]
            tril = _hstack_datasets([dat_ij[str(i)] for i in range(len(dat_ij))], slices)
            triu = _hstack_datasets([dat_ji[str(i)] for i in range(len(dat_ji))], slices)
            assert tril.dtype == numpy.complex128
            naux = self.naux
            nao = self.nao
            out = numpy.empty((naux, nao*nao), dtype=tril.dtype)
            libpbc.PBCunpack_tril_triu(out.ctypes.data_as(ctypes.c_void_p),
                                       tril.ctypes.data_as(ctypes.c_void_p),
                                       triu.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(naux), ctypes.c_int(nao))
        return out

    def load(self, kpti, kptj):
        if self._data_version == 'v1':
            with _load3c(self._cderi, self._label) as fload:
                return numpy.asarray(fload(kpti, kptj))

        ki = member(kpti, self.kpts)
        kj = member(kptj, self.kpts)
        if len(ki) == 0 or len(kj) == 0:
            raise RuntimeError(f'CDERI for kpts ({kpti}, {kptj}) not found')
        return self._load_one(ki[0], kj[0], ())

    def __array__(self):
        '''Create a numpy array'''
        return self[:]

class _load3c:
    #'''Read cderi from old version pyscf (<= 2.0)'''
    '''cderi file may be stored in different formats (version 1 generated from
    pyscf-2.0 or older, version 2 from pyscf-2.1 or newer). This function
    can read both data formats.
    '''
    def __init__(self, cderi, label, kpti_kptj=None, kptij_label='j3c-kptij',
                 ignore_key_error=False):
        self.cderi = cderi
        self.label = label
        self.kptij_label = kptij_label
        self.kpti_kptj = kpti_kptj
        self.ignore_key_error = ignore_key_error
        self.feri = None
        self._kptij_lst = None
        self._aosym = None

    def __enter__(self):
        self.feri = h5py.File(self.cderi, 'r')
        if self.label not in self.feri:
            # Return a size-0 array to skip the loop in sr_loop
            if self.ignore_key_error:
                return numpy.zeros(0)
            else:
                raise KeyError('Key "%s" not found' % self.label)

        if self.kpti_kptj is None:
            return self.getitem
        else:
            return self.getitem(*self.kpti_kptj)

    def __exit__(self, type, value, traceback):
        self.feri.close()

    @property
    def kptij_lst(self):
        if self._kptij_lst is None:
            if self.data_version == 'v2':
                self._kptij_lst = self.feri['kpts'][()]
            else:
                self._kptij_lst = self.feri[self.kptij_label][()]
        return self._kptij_lst

    @property
    def aosym(self):
        if self._aosym is None:
            self._aosym = self.feri['aosym'][()]
            if isinstance(self._aosym, bytes):
                self._aosym = self._aosym.decode()
        return self._aosym

    @property
    def data_version(self):
        '''Guess the data format version'''
        if 'kpts' in self.feri:
            return 'v2'
        else:
            return 'v1'

    def getitem(self, kpti, kptj):
        assert self.feri is not None
        if self.data_version == 'v2':
            kpts = self.kptij_lst
            nkpts = len(kpts)
            if (isinstance(kpti, (int, numpy.integer)) and
                isinstance(kptj, (int, numpy.integer))):
                ki = kpti
                kj = kptj
            else:
                ki = member(kpti, kpts)
                kj = member(kptj, kpts)
                if len(ki) == 0 or len(kj) == 0:
                    raise RuntimeError(f'CDERI {self.label} for kpts ({kpti}, {kptj}) is '
                                       'not initialized.')

                ki = ki[0]
                kj = kj[0]

            key = f'{self.label}/{ki * nkpts + kj}'
            if key not in self.feri:
                if self.ignore_key_error:
                    return numpy.zeros(0)
                else:
                    raise KeyError(f'Key {key} not found')
            return _KPair3CLoader(self.feri[self.label], ki, kj, nkpts, self.aosym)

        else:  # data format version 1
            return _getitem(self.feri, self.label, (kpti, kptj), self.kptij_lst,
                            self.ignore_key_error)

def _getitem(h5group, label, kpti_kptj, kptij_lst, ignore_key_error=False,
             aosym=None):
    kpti_kptj = numpy.asarray(kpti_kptj)
    k_id = member(kpti_kptj, kptij_lst)
    if len(k_id) > 0:
        key = label + '/' + str(k_id[0])
        if key not in h5group:
            if ignore_key_error:
                return numpy.zeros(0)
            else:
                raise KeyError('Key "%s" not found' % key)
        hermi = False
    else:
        # swap ki,kj due to the hermiticity
        kptji = kpti_kptj[[1,0]]
        k_id = member(kptji, kptij_lst)
        if len(k_id) == 0:
            raise RuntimeError('%s for kpts %s is not initialized.\n'
                               'You need to update the attribute .kpts then call '
                               '.build() to initialize %s.'
                               % (label, kpti_kptj, label))

        key = label + '/' + str(k_id[0])
        if key not in h5group:
            if ignore_key_error:
                return numpy.zeros(0)
            else:
                raise KeyError('Key "%s" not found' % key)
        hermi = True

    dat = _load_and_unpack(h5group[key],hermi)
    return dat

class _load_and_unpack:
    '''
    This class returns an array-like object to an hdf5 file that can
    be sliced, to allow for lazy loading

    hermi : boolean
    Take the conjugate transpose of the slice

    See PR 1086 and Issue 1076
    '''
    def __init__(self, dat, hermi):
        self.dat = dat
        self.hermi = hermi
    def __getitem__(self, s):
        dat = self.dat
        if isinstance(dat, h5py.Group):
            v = _hstack_datasets([dat[str(i)] for i in range(len(dat))], s)
        else: # For mpi4pyscf, pyscf-1.5.1 or older
            v = numpy.asarray(dat[s])

        if self.hermi:
            nao = int(numpy.sqrt(v.shape[-1]))
            v1 = lib.transpose(v.reshape(-1,nao,nao), axes=(0,2,1)).conj()
            return v1.reshape(v.shape)
        else:
            return v
    def __array__(self):
        '''Create a numpy array'''
        return self[()]

    @property
    def shape(self):
        dat = self.dat
        if isinstance(dat, h5py.Group):
            all_shape = [dat[str(i)].shape for i in range(len(dat))]
            shape = all_shape[0][:-1] + (sum(x[-1] for x in all_shape),)
            return shape
        else: # For mpi4pyscf, pyscf-1.5.1 or older
            return dat.shape

def _hstack_datasets(data_to_stack, slices=numpy.s_[:]):
    """Faster version of the operation
    np.hstack([x[slices] for x in data_to_stack]) for h5py datasets.

    Parameters
    ----------
    data_to_stack : list of h5py.Dataset or np.ndarray
        Datasets/arrays to be stacked along first axis.
    slices: tuple or list of slices, a slice, or ().
        The slices (or indices) to select data from each H5 dataset.

    Returns
    -------
    numpy.ndarray
        The stacked data, equal to numpy.hstack([dset[slices] for dset in data_to_stack])
    """
    # Step 1. Calculate the shape of the output array, and store it
    # in res_shape.
    res_shape = list(data_to_stack[0].shape)
    dset_shapes = [x.shape for x in data_to_stack]

    if not (isinstance(slices, tuple) or isinstance(slices, list)):
        # If slices is not a tuple, we assume it is a single slice acting on axis 0 only.
        slices = (slices,)

    def len_of_slice(arraylen, s):
        start, stop, step = s.indices(arraylen)
        r = range(start, stop, step)
        # Python has a very fast builtin method to get the length of a range.
        return len(r)

    for i, cur_slice in enumerate(slices):
        if not isinstance(cur_slice, slice):
            return numpy.hstack([dset[slices] for dset in data_to_stack])
        if i == 1:
            ax1widths_sliced = [len_of_slice(shp[1], cur_slice) for shp in dset_shapes]
        else:
            # Except along axis 1, we assume the dimensions of all datasets are the same.
            # If they aren't, an error gets raised later.
            res_shape[i] = len_of_slice(res_shape[i], cur_slice)
    if len(slices) <= 1:
        ax1widths_sliced = [shp[1] for shp in dset_shapes]

    # Final dim along axis 1 is the sum of the post-slice axis 1 widths.
    res_shape[1] = sum(ax1widths_sliced)

    # Step 2. Allocate the output buffer
    out = numpy.empty(res_shape, dtype=numpy.result_type(*[dset.dtype for dset in data_to_stack]))

    # Step 3. Read data into the output buffer.
    ax1ind = 0
    for i, dset in enumerate(data_to_stack):
        ax1width = ax1widths_sliced[i]
        dest_sel = numpy.s_[:, ax1ind:ax1ind + ax1width]
        if hasattr(dset, 'read_direct'):
            # h5py has issues with zero-size selections, see
            # https://github.com/h5py/h5py/issues/1455,
            # so we check for that here.
            if out[dest_sel].size > 0:
                dset.read_direct(
                    out,
                    source_sel=slices,
                    dest_sel=dest_sel
                )
        else:
            # For array-like objects
            out[dest_sel] = dset[slices]
        ax1ind += ax1width
    return out

class _KPair3CLoader:
    def __init__(self, dat, ki, kj, nkpts, aosym):
        self.dat = dat
        self.kikj = ki * nkpts + kj
        self.kjki = kj * nkpts + ki
        self.nkpts = nkpts
        self.nsegs = len(dat[str(self.kikj)])
        self.aosym = aosym

    def __getitem__(self, s):
        if self.aosym == 's1' or self.kikj == self.kjki:
            dat = self.dat[str(self.kikj)]
            out = _hstack_datasets([dat[str(i)] for i in range(self.nsegs)], s)
        elif self.aosym == 's2':
            dat_ij = self.dat[str(self.kikj)]
            dat_ji = self.dat[str(self.kjki)]
            tril = _hstack_datasets([dat_ij[str(i)] for i in range(self.nsegs)], s)
            triu = _hstack_datasets([dat_ji[str(i)] for i in range(self.nsegs)], s)
            assert tril.dtype == numpy.complex128
            naux, nao_pair = tril.shape
            nao = int((nao_pair * 2)**.5)
            out = numpy.empty((naux, nao*nao), dtype=tril.dtype)
            libpbc.PBCunpack_tril_triu(out.ctypes.data_as(ctypes.c_void_p),
                                       tril.ctypes.data_as(ctypes.c_void_p),
                                       triu.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(naux), ctypes.c_int(nao))
        else:
            raise ValueError(f'Unknown aosym {self.aosym}')
        return out

    def __array__(self):
        '''Create a numpy array'''
        return self[()]

    @property
    def shape(self):
        dat = self.dat[str(self.kikj)]
        shapes = [dat[str(i)].shape for i in range(self.nsegs)]
        naux = shapes[0][0]
        nao_pair = sum([shape[1] for shape in shapes])
        if self.aosym == 's1' or self.kikj == self.kjki:
            return (naux, nao_pair)
        else:
            nao = int((nao_pair * 2)**.5)
            return (naux, nao*nao)

def _gaussian_int(cell):
    r'''Regular gaussian integral \int g(r) dr^3'''
    return ft_ao.ft_ao(cell, numpy.zeros((1,3)))[0].real
