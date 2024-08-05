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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
J-metric density fitting
'''


import tempfile
import contextlib
import numpy
import h5py
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.df import incore
from pyscf.df import outcore
from pyscf.df import r_incore
from pyscf.df import addons
from pyscf.df import df_jk
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos, iden_coeffs
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf import __config__

class DF(lib.StreamObject):
    r'''
    Object to hold 3-index tensor

    Attributes:
        auxbasis : str or dict
            Same input format as :attr:`Mole.basis`

        auxmol : Mole object
            Read only Mole object to hold the auxiliary basis.  auxmol is
            generated automatically in the initialization step based on the
            given auxbasis.  It is used in the rest part of the code to
            determine the problem size, the integral batches etc.  This object
            should NOT be modified.
        _cderi_to_save : str
            If _cderi_to_save is specified, the DF integral tensor will be
            saved in this file.
        _cderi : str or numpy array
            If _cderi is specified, the DF integral tensor will be read from
            this HDF5 file (or numpy array). When the DF integral tensor is
            provided from the HDF5 file, its dataset name should be consistent
            with DF._dataname, which is 'j3c' by default.
            The DF integral tensor :math:`V_{x,ij}` should be a 2D array in C
            (row-major) convention, where x corresponds to index of auxiliary
            basis, and the composite index ij is the orbital pair index. The
            hermitian symmetry is assumed for the composite ij index, ie
            the elements of :math:`V_{x,i,j}` with :math:`i\geq j` are existed
            in the DF integral tensor.  Thus the shape of DF integral tensor
            is (M,N*(N+1)/2), where M is the number of auxbasis functions and
            N is the number of basis functions of the orbital basis.
        blockdim : int
            When reading DF integrals from disk the chunk size to load.  It is
            used to improve IO performance.
    '''

    blockdim = getattr(__config__, 'df_df_DF_blockdim', 240)

    # Store DF tensor in a format compatible to pyscf-1.1 - pyscf-1.6
    _compatible_format = getattr(__config__, 'df_df_DF_compatible_format', False)
    _dataname = 'j3c'

    _keys = {'mol', 'auxmol'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self._auxbasis = auxbasis

##################################################
# Following are not input options
        self.auxmol = None
# If _cderi_to_save is specified, the 3C-integral tensor will be saved in this file.
        self._cderi_to_save = None
# If _cderi is specified, the 3C-integral tensor will be read from this file
        self._cderi = None
        self._vjopt = None
        self._rsh_df = {}  # Range separated Coulomb DF objects

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('_cderi_to_save', '_cderi', '_vjopt', '_rsh_df'),
            reset_state=True)

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self.reset()
            self._auxbasis = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        if self.auxmol is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = auxmol.basis = %s', self.auxmol.basis)
        log.info('max_memory = %s', self.max_memory)
        if isinstance(self._cderi, str):
            log.info('_cderi = %s  where DF integrals are loaded (readonly).',
                     self._cderi)
        return self

    def build(self):
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        self.check_sanity()
        self.dump_flags()
        if self._cderi is not None and self.auxmol is None:
            log.info('Skip DF.build(). Tensor _cderi will be used.')
            return self

        mol = self.mol
        auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nao_pair = nao*(nao+1)//2

        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if (nao_pair*naux*8/1e6 < .9*max_memory and not is_custom_storage):
            self._cderi = incore.cholesky_eri(mol, int3c=int3c, int2c=int2c,
                                              auxmol=auxmol,
                                              max_memory=max_memory, verbose=log)
        else:
            if self._cderi_to_save is None:
                self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            cderi = self._cderi_to_save

            if is_custom_storage:
                cderi_name = cderi
            else:
                cderi_name = cderi.name
            if self._cderi is None:
                log.info('_cderi_to_save = %s', cderi_name)
            else:
                # If cderi needs to be saved in
                log.warn('Value of _cderi is ignored. DF integrals will be '
                         'saved in file %s .', cderi_name)

            if self._compatible_format:
                outcore.cholesky_eri(mol, cderi, dataname=self._dataname,
                                     int3c=int3c, int2c=int2c, auxmol=auxmol,
                                     max_memory=max_memory, verbose=log)
            else:
                # Store DF tensor in blocks. This is to reduce the
                # initialization overhead
                outcore.cholesky_eri_b(mol, cderi, dataname=self._dataname,
                                       int3c=int3c, int2c=int2c, auxmol=auxmol,
                                       max_memory=max_memory, verbose=log)
            self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)
        return self

    def kernel(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
            self.auxmol = None
        self._cderi = None
        self._vjopt = None
        self._rsh_df = {}
        return self

    def loop(self, blksize=None):
        if self._cderi is None:
            self.build()
        if blksize is None:
            blksize = self.blockdim

        with addons.load(self._cderi, self._dataname) as feri:
            if isinstance(feri, numpy.ndarray):
                naoaux = feri.shape[0]
                for b0, b1 in self.prange(0, naoaux, blksize):
                    yield numpy.asarray(feri[b0:b1], order='C')

            else:
                if isinstance(feri, h5py.Group):
                    # starting from pyscf-1.7, DF tensor may be stored in
                    # block format
                    naoaux = feri['0'].shape[0]
                    def load(aux_slice):
                        b0, b1 = aux_slice
                        return _load_from_h5g(feri, b0, b1)
                else:
                    naoaux = feri.shape[0]
                    def load(aux_slice):
                        b0, b1 = aux_slice
                        return numpy.asarray(feri[b0:b1])

                for dat in lib.map_with_prefetch(load, self.prange(0, naoaux, blksize)):
                    yield dat
                    dat = None

    def prange(self, start, end, step):
        for i in range(start, end, step):
            yield i, min(i+step, end)

    def get_naoaux(self):
        # determine naoaux with self._cderi, because DF object may be used as CD
        # object when self._cderi is provided.
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi, self._dataname) as feri:
            if isinstance(feri, h5py.Group):
                return feri['0'].shape[0]
            else:
                return feri.shape[0]

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is None:
            return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)

        # A temporary treatment for RSH-DF integrals
        with self.range_coulomb(omega) as rsh_df:
            return df_jk.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

    def get_eri(self):
        nao = self.mol.nao_nr()
        nao_pair = nao * (nao+1) // 2
        ao_eri = numpy.zeros((nao_pair,nao_pair))
        for eri1 in self.loop():
            lib.dot(eri1.T, eri1, 1, ao_eri, 1)
        return ao2mo.restore(8, ao_eri, nao)
    get_ao_eri = get_eri

    def ao2mo(self, mo_coeffs,
              compact=getattr(__config__, 'df_df_DF_ao2mo_compact', True)):
        if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
            mo_coeffs = (mo_coeffs,) * 4
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        mo_eri = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        Lij = Lkl = None
        for eri1 in self.loop():
            Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=Lij)
            if sym:
                Lkl = Lij
            else:
                Lkl = _ao2mo.nr_e2(eri1, mokl, klslice, aosym='s2', mosym=klmosym, out=Lkl)
            lib.dot(Lij.T, Lkl, 1, mo_eri, 1)
        return mo_eri
    get_mo_eri = ao2mo

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        '''Creates a temporary density fitting object for RSH-DF integrals.
        In this context, only LR or SR integrals for mol and auxmol are computed.
        '''
        if omega is None:
            omega = 0
        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            if hasattr(self, '_dataname'):
                rsh_df._dataname = f'{self._dataname}-lr/{key}'
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        mol = self.mol
        auxmol = self.auxmol

        mol_omega = mol.omega
        mol.omega = omega
        auxmol_omega = None
        if auxmol is not None:
            auxmol_omega = auxmol.omega
            auxmol.omega = omega

        assert rsh_df.mol.omega == omega
        if rsh_df.auxmol is not None:
            assert rsh_df.auxmol.omega == omega

        try:
            yield rsh_df
        finally:
            mol.omega = mol_omega
            if auxmol_omega is not None:
                auxmol.omega = auxmol_omega

    to_gpu = lib.to_gpu

GDF = DF


class DF4C(DF):
    '''Relativistic 4-component'''
    def build(self):
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol
        auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
        n2c = mol.nao_2c()
        naux = auxmol.nao_nr()
        nao_pair = n2c*(n2c+1)//2

        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        if nao_pair*naux*3*16/1e6*2 < max_memory:
            self._cderi =(r_incore.cholesky_eri(mol, auxmol=auxmol, aosym='s2',
                                                int3c='int3c2e_spinor', verbose=log),
                          r_incore.cholesky_eri(mol, auxmol=auxmol, aosym='s2',
                                                int3c='int3c2e_spsp1_spinor', verbose=log))
        else:
            raise NotImplementedError
        return self

    def loop(self, blksize=None):
        if self._cderi is None:
            self.build()
        if blksize is None:
            blksize = self.blockdim
        with addons.load(self._cderi[0], self._dataname) as ferill:
            naoaux = ferill.shape[0]
            with addons.load(self._cderi[1], self._dataname) as feriss: # python2.6 not support multiple with
                for b0, b1 in self.prange(0, naoaux, blksize):
                    erill = numpy.asarray(ferill[b0:b1], order='C')
                    eriss = numpy.asarray(feriss[b0:b1], order='C')
                    yield erill, eriss

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is None:
            return df_jk.r_get_jk(self, dm, hermi, with_j, with_k)

        with self.range_coulomb(omega) as rsh_df:
            return df_jk.r_get_jk(rsh_df, dm, hermi, with_j, with_k)

    def get_eri(self):
        raise NotImplementedError
    get_ao_eri = get_eri

    def ao2mo(self, mo_coeffs):
        raise NotImplementedError
    get_mo_eri = ao2mo

GDF4C = DF4C
