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

class DF(lib.StreamObject):
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.auxbasis = 'weigend+etb'
        self.auxmol = None
        self._cderi_file = tempfile.NamedTemporaryFile()
        self._cderi = None
        self._call_count = 0
        self.blockdim = 240

    def build(self):
        t0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        mol = self.mol
        auxmol = self.auxmol = incore.format_aux_basis(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nao_pair = nao*(nao+1)//2

        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        if nao_pair*nao*3*8/1e6 < max_memory:
            self._cderi = incore.cholesky_eri(mol, auxmol=auxmol, verbose=log)
        else:
            if not isinstance(self._cderi, str):
                if isinstance(self._cderi_file, str):
                    self._cderi = self._cderi_file
                else:
                    self._cderi = self._cderi_file.name
            outcore.cholesky_eri(mol, self._cderi, auxmol=auxmol, verbose=log)
            if nao_pair*nao*8/1e6 < max_memory:
                with addons.load(self._cderi) as feri:
                    cderi = numpy.asarray(feri)
                self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)

        return self

    def loop(self):
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi) as feri:
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
        with addons.load(self._cderi) as feri:
            return feri.shape[0]

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        from pyscf.df import df_jk
        return df_jk.get_jk(self, dm, hermi, vhfopt, with_j, with_k)

    def ao2mo(self, mo_coeffs):
        from pyscf.ao2mo import _ao2mo
        nmoi, nmoj, nmok, nmol = [x.shape[1] for x in mo_coeffs]
        mo_eri = numpy.zeros((nmoi*nmoj,nmok*nmol))
        moij = numpy.asarray(numpy.hstack((mo_coeffs[0],mo_coeffs[1])), order='F')
        ijshape = (0, nmoi, nmoi, nmoi+nmoj)
        mokl = numpy.asarray(numpy.hstack((mo_coeffs[2],mo_coeffs[3])), order='F')
        klshape = (0, nmok, nmok, nmok+nmol)
        for eri1 in self.loop():
            buf1 = _ao2mo.nr_e2(eri1, moij, ijshape, 's2', 's1')
            buf2 = _ao2mo.nr_e2(eri1, mokl, klshape, 's2', 's1')
            lib.dot(buf1.T, buf2, 1, mo_eri, 1)
        return mo_eri

    def update_mf(self, mf):
        from pyscf.df import df_jk
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
            self._cderi = r_incore.cholesky_eri(mol, auxbasis=self.auxbasis,
                                                aosym='s2', verbose=log)
        else:
            raise NotImplementedError
            self._cderifile = self._cderi
            self._cderi = r_outcore.cholesky_eri(mol, self._cderi.name,
                                                 auxbasis=self.auxbasis,
                                                 verbose=log)
        return self

    def loop(self):
        if self._cderi is None:
            self.build()
        with addons.load(self._cderi[0]) as ferill:
            naoaux = ferill.shape[0]
            with addons.load(self._cderi[1]) as feriss: # python2.6 not support multiple with
                for b0, b1 in self.prange(0, naoaux, self.blockdim):
                    erill = numpy.asarray(ferill[b0:b1], order='C')
                    eriss = numpy.asarray(feriss[b0:b1], order='C')
                    yield erill, eriss

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        from pyscf.df import df_jk
        return df_jk.r_get_jk(self, dm, hermi)

    def ao2mo(self, mo_coeffs):
        pass

