# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
# Author: Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''

import time
#import ctypes
#from functools import reduce
import numpy
#from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
#from pyscf.ao2mo import _ao2mo
#from pyscf.cc import _ccsd
#from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core
from pyscf import __config__
#
#BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
#MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def kernel(myadc, eris, verbose=None):

    log = logger.new_logger(myadc, verbose)
    if eris is None:
        #eris = mycc.ao2mo(mycc.mo_coeff)
        # TODO: transform integrals if they are not provided
        raise NotImplementedError('Integrals for UADC amplitudes')

    cput0 = (time.clock(), time.time())

    t1, t2 = myadc.compute_amplitudes(eris)
    e_corr = myadc.energy(t1, t2, eris)

    log.info('E(corr) = %.15g', e_corr)
    log.timer('ADC ground-state energy', *cput0)
    return e_corr, t1, t2


class UADC(lib.StreamObject):

    incore_complete = getattr(__config__, 'adc_uadc_UADC_incore_complete', False)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf import gto

        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.mol.nelectron
        self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        self.chkfile = mf.chkfile

    def kernel(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        # TODO: Implement
        #self.dump_flags()

        # TODO: ao2mo transformation if eris is None
        eris = uadc_ao2mo.transform_integrals(self)
        exit()

        self.e_corr, self.t1, self.t2 = kernel(self, eris, verbose=self.verbose)

        # TODO: Implement
        #self._finalize()
        return self.e_corr, self.t1, self.t2

# TODO: add a test main section
