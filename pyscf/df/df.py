#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
J-metric density fitting
'''

import time
import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df import incore
from pyscf.df import outcore
from pyscf.df import r_incore
from pyscf.df import addons
from pyscf.df import df_jk
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos, iden_coeffs

class DF(lib.StreamObject):
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.auxbasis = 'weigend+etb'
        self.auxmol = None
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self._cderi = None
        self._call_count = 0
        self.blockdim = 240
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'auxbasis = %s', self.auxbasis)
        logger.info(self, 'max_memory = %s', self.max_memory)
        if isinstance(self._cderi, str):
            logger.info(self, '_cderi = %s  where DF integrals are loaded (readonly).',
                        self._cderi)
        elif isinstance(self._cderi_to_save, str):
            logger.info(self, '_cderi_to_save = %s', self._cderi_to_save)
        else:
            logger.info(self, '_cderi_to_save = %s', self._cderi_to_save.name)

    def build(self):
        t0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol
        auxmol = self.auxmol = incore.format_aux_basis(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nao_pair = nao*(nao+1)//2

        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if (nao_pair*naux*3*8/1e6 < max_memory and
            not isinstance(self._cderi_to_save, str)):
            self._cderi = incore.cholesky_eri(mol, int3c=int3c, int2c=int2c,
                                              auxmol=auxmol, verbose=log)
        else:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                log.warn('Value of _cderi is ignored. DF integrals will be '
                         'saved in file %s .', cderi)
            outcore.cholesky_eri(mol, cderi, dataname='j3c',
                                 int3c=int3c, int2c=int2c, auxmol=auxmol,
                                 max_memory=max_memory, verbose=log)
            if nao_pair*naux*8/1e6 < max_memory:
                with addons.load(cderi, 'j3c') as feri:
                    cderi = numpy.asarray(feri)
            self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)
        return self

    def loop(self):
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi, 'j3c') as feri:
            naoaux = feri.shape[0]
            for b0, b1 in self.prange(0, naoaux, self.blockdim):
                eri1 = numpy.asarray(feri[b0:b1], order='C')
                yield eri1

    def prange(self, start, end, step):
        self._call_count += 1
        if self._call_count % 2 == 1:
            for i in reversed(range(start, end, step)):
                yield i, min(i+step, end)
        else:
            for i in range(start, end, step):
                yield i, min(i+step, end)

    def get_naoaux(self):
# determine naoaux with self._cderi, because DF object may be used as CD
# object when self._cderi is provided.
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi, 'j3c') as feri:
            return feri.shape[0]

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        return df_jk.get_jk(self, dm, hermi, vhfopt, with_j, with_k)

    def get_eri(self):
        nao = self.mol.nao_nr()
        nao_pair = nao * (nao+1) // 2
        ao_eri = numpy.zeros((nao_pair,nao_pair))
        for eri1 in self.loop():
            lib.dot(eri1.T, eri1, 1, ao_eri, 1)
        return ao2mo.restore(8, ao_eri, nao)
    get_ao_eri = get_eri

    def ao2mo(self, mo_coeffs, compact=True):
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

    def update_mf(self, mf):
        return df_jk.density_fit(mf, self.auxbasis, self)

    def update_mc(self, mc):
        from pyscf.mcscf import df
        return df.density_fit(mc, self.auxbasis, self)

    def update_mp2(self):
        pass

    def update(self):
        pass


class DF4C(DF):
    '''Relativistic 4-component'''
    def build(self):
        t0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol
        auxmol = self.auxmol = incore.format_aux_basis(self.mol, self.auxbasis)
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

    def loop(self):
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi[0], 'j3c') as ferill:
            naoaux = ferill.shape[0]
            with addons.load(self._cderi[1], 'j3c') as feriss: # python2.6 not support multiple with
                for b0, b1 in self.prange(0, naoaux, self.blockdim):
                    erill = numpy.asarray(ferill[b0:b1], order='C')
                    eriss = numpy.asarray(feriss[b0:b1], order='C')
                    yield erill, eriss

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        return df_jk.r_get_jk(self, dm, hermi)

    def ao2mo(self, mo_coeffs):
        pass

