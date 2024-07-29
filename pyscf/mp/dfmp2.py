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

'''
density fitting MP2,  3-center integrals incore.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
from pyscf.mp.mp2 import make_rdm1, make_rdm2
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
    else:
        t2 = None

    Lov = numpy.empty((naux, nocc*nvir))
    p1 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        p0, p1 = p1, p1 + qov.shape[0]
        Lov[p0:p1] = qov

    emp2_ss = emp2_os = 0

    for i in range(nocc):
        buf = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                        Lov).reshape(nvir,nocc,nvir)
        gi = numpy.array(buf, copy=False)
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
        edi = numpy.einsum('jab,jab', t2i, gi) * 2
        exi = -numpy.einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i
        buf = gi = t2i = None # free mem

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFMP2(mp2.MP2):
    _keys = set(['with_df'])

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def loop_ao2mo(self, mo_coeff, nocc):
        mo = numpy.asarray(mo_coeff, order='F')
        nmo = mo.shape[1]
        ijslice = (0, nocc, nocc, nmo)
        Lov = None
        with_df = self.with_df

        nvir = nmo - nocc
        naux = with_df.get_naoaux()
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        blksize = int(min(naux, max(with_df.blockdim,
                                    (max_memory*1e6/8-nocc*nvir**2*2)/(nocc*nvir))))
        for eri1 in with_df.loop(blksize=blksize):
            Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
            yield Lov

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        # Initialize only the mo_coeff
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

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = DFMP2

from pyscf import scf
scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)
scf.rohf.ROHF.DFMP2 = None
scf.uhf.UHF.DFMP2 = None

del (WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254500453)

    mf = scf.density_fit(scf.RHF(mol), 'weigend')
    mf.kernel()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203738031827)

    pt.frozen = 2
    emp2, t2 = pt.kernel()
    print(emp2 - -0.14433975122418313)
