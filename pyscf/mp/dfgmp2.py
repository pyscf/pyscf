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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
density fitting GMP2,  3-center integrals incore.
'''

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.mp import gmp2, dfmp2
from pyscf.mp.gmp2 import make_rdm1, make_rdm2

WITH_T2 = getattr(__config__, 'mp_dfgmp2_with_t2', True)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = np.empty((nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
    else:
        t2 = None

    orbspin = eris.orbspin
    Lova = np.empty((naux, nocc*nvir))
    if orbspin is None:
        Lovb = np.empty((naux, nocc*nvir))

    p1 = 0
    for qova, qovb in mp.loop_ao2mo(mo_coeff, nocc, orbspin):
        p0, p1 = p1, p1 + qova.shape[0]
        Lova[p0:p1] = qova
        if orbspin is None:
            Lovb[p0:p1] = qovb

    if orbspin is not None:
        sym_forbid = (orbspin[:nocc,None] != orbspin[nocc:]).flatten()
        Lova[:,sym_forbid] = 0

    emp2 = 0
    for i in range(nocc):
        if orbspin is None:
            buf  = np.dot(Lova[:,i*nvir:(i+1)*nvir].T, Lova)
            buf += np.dot(Lovb[:,i*nvir:(i+1)*nvir].T, Lovb)
            buf += np.dot(Lova[:,i*nvir:(i+1)*nvir].T, Lovb)
            buf += np.dot(Lovb[:,i*nvir:(i+1)*nvir].T, Lova)
        else:
            buf = np.dot(Lova[:,i*nvir:(i+1)*nvir].T, Lova)
        gi = np.array(buf, copy=False).reshape(nvir,nocc,nvir)
        gi = gi.transpose(1,0,2) - gi.transpose(1,2,0)
        t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
        emp2 += np.einsum('jab,jab', t2i, gi) * .25
        if with_t2:
            t2[i] = t2i

    return emp2, t2


class DFGMP2(dfmp2.DFMP2):
    def loop_ao2mo(self, mo_coeff, nocc, orbspin):
        nao, nmo = mo_coeff.shape
        if orbspin is None:
            moa = np.asarray(mo_coeff[:nao//2], order='F')
            mob = np.asarray(mo_coeff[nao//2:], order='F')
        else:
            moa = np.asarray(mo_coeff[:nao//2]+mo_coeff[nao//2:], order='F')

        ijslice = (0, nocc, nocc, nmo)
        Lova = Lovb = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        if orbspin is None:
            fac = 2
        else:
            fac = 1
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(fac*nocc*nvir))))
        if orbspin is None:
            for eri1 in with_df.loop(blksize=blksize):
                Lova = _ao2mo.nr_e2(eri1, moa, ijslice, aosym='s2', out=Lova)
                Lovb = _ao2mo.nr_e2(eri1, mob, ijslice, aosym='s2', out=Lovb)
                yield Lova, Lovb
        else:
            for eri1 in with_df.loop(blksize=blksize):
                Lova = _ao2mo.nr_e2(eri1, moa, ijslice, aosym='s2', out=Lova)
                yield Lova, None

    def ao2mo(self, mo_coeff=None):
        eris = gmp2._PhysicistsERIs()
        # Initialize only the mo_coeff and
        eris._common_init_(self, mo_coeff)
        return eris

    def make_rdm1(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm1(self, t2, ao_repr=ao_repr)

    def make_rdm2(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm2(self, t2, ao_repr=ao_repr)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)


if __name__ == "__main__":
    from pyscf import gto, scf, mp
    mol = gto.Mole()
    mol.atom = [
        ['Li', (0., 0., 0.)],
        ['H', (1., 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.build()

    mf = scf.GHF(mol).run()
    mymp = DFGMP2(mf)
    mymp.kernel()

    mf = scf.RHF(mol).run()
    mf = mf.to_ghf()
    mymp = DFGMP2(mf)
    mymp.kernel()

    mymp = mp.GMP2(mf).density_fit()
    mymp.kernel()

    mf = scf.RHF(mol).density_fit().run()
    mymp = mp.GMP2(mf)
    mymp.kernel()
