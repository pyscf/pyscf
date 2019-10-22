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

#import time
#import ctypes
#from functools import reduce
#import numpy
#from pyscf import gto
#from pyscf import lib
#from pyscf.lib import logger
#from pyscf import ao2mo
#from pyscf.ao2mo import _ao2mo
#from pyscf.cc import _ccsd
#from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core
#from pyscf import __config__
#
#BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
#MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


#def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
#           tolnormt=1e-6, verbose=None):
#    log = logger.new_logger(mycc, verbose)
#    if eris is None:
#        eris = mycc.ao2mo(mycc.mo_coeff)
#    if t1 is None and t2 is None:
#        t1, t2 = mycc.get_init_guess(eris)
#    elif t2 is None:
#        t2 = mycc.get_init_guess(eris)[1]
#
#    cput1 = cput0 = (time.clock(), time.time())
#    eold = 0
#    eccsd = mycc.energy(t1, t2, eris)
#    log.info('Init E(CCSD) = %.15g', eccsd)
#
#    if isinstance(mycc.diis, lib.diis.DIIS):
#        adiis = mycc.diis
#    elif mycc.diis:
#        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
#        adiis.space = mycc.diis_space
#    else:
#        adiis = None
#
#    conv = False
#    for istep in range(max_cycle):
#        t1new, t2new = mycc.update_amps(t1, t2, eris)
#        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
#        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
#        normt = numpy.linalg.norm(tmpvec)
#        tmpvec = None
#        if mycc.iterative_damping < 1.0:
#            alpha = mycc.iterative_damping
#            t1new = (1-alpha) * t1 + alpha * t1new
#            t2new *= alpha
#            t2new += (1-alpha) * t2
#        t1, t2 = t1new, t2new
#        t1new = t2new = None
#        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
#        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
#        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
#                 istep+1, eccsd, eccsd - eold, normt)
#        cput1 = log.timer('CCSD iter', *cput1)
#        if abs(eccsd-eold) < tol and normt < tolnormt:
#            conv = True
#            break
#    log.timer('CCSD', *cput0)
#    return conv, eccsd, t1, t2


class UADC(lib.StreamObject):

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
        self._nocc = None
        self._nmo = None
        self.chkfile = mf.chkfile

# TODO: add a test main section
