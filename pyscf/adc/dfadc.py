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

import time
import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.adc import radc
from pyscf import __config__

class RADC(radc.RADC):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        radc.RADC.__init__(self, mf, frozen, mo_coeff, mo_occ)

#        if getattr(mf, 'with_df', None):
#            self.with_df = mf.with_df
#        else:
#            self.with_df = df.DF(mf.mol)
#            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
#        self._keys.update(['with_df'])
#
#    def reset(self, mol=None):
#        self.with_df.reset(mol)
#        return radc.RADC.reset(self, mol)
#
#    def ao2mo(self, mo_coeff=None):
#        return _make_df_eris(self, mo_coeff)


#def _make_df_eris(adc, mo_coeff=None):
#
#    cput0 = (time.clock(), time.time())
#    log = logger.Logger(mycc.stdout, mycc.verbose)
#
#    mo_coeff = np.asarray(eris.mo_coeff, order='F')
#    nocc = myadc._nocc
#    nao, nmo = mo_coeff.shape
#    nvir = myadc._nmo - myadc._nocc
#    nvir_pair = nvir*(nvir+1)//2
#
#    naux = myadc._scf.with_df.get_naoaux()
#    Loo = np.empty((naux,nocc,nocc))
#    Lov = np.empty((naux,nocc,nvir))
#    Lvo = np.empty((naux,nvir,nocc))
#    Lvv = np.empty((naux,nvir_pair))
#    ijslice = (0, nmo, 0, nmo)
#    Lpq = None
#    p1 = 0
#    for eri1 in myadc._scf.with_df.loop():
#        Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
#        p0, p1 = p1, p1 + Lpq.shape[0]
#        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
#        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
#        Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
#        Lvv[p0:p1] = lib.pack_tril(Lpq[:,nocc:,nocc:].reshape(-1,nvir,nvir))
#    Loo = Loo.reshape(naux,nocc*nocc)
#    Lov = Lov.reshape(naux,nocc*nvir)
#    Lvo = Lvo.reshape(naux,nocc*nvir)
#
#    eris.feri1 = lib.H5TmpFile()
#    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
#    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
#    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
#    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
#    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
#    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
#    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
#    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
#    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
#    eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv)).reshape(nocc,nocc,nvir,nvir)
#    eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocc,nvir,nvir,nocc)
#    eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
#    eris.ovvv[:] = lib.ddot(Lov.T, Lvv).reshape(nocc,nvir,nvir_pair)
#    eris.vvvv[:] = lib.ddot(Lvv.T, Lvv)
#    log.timer('CCSD integral transformation', *cput0)
#
#    return eris

