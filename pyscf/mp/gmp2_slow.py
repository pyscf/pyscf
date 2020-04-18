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
GMP2 implementation which supports both real and complex orbitals
GMP2 in spin-orbital form
E(MP2) = 1/4 <ij||ab><ab||ij>/(ei+ej-ea-eb)
'''

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import gmp2
from pyscf import scf
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_gmp2_with_t2', True)

class GMP2(gmp2.GMP2):
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = nocc**2*nvir**2*3 * 16/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff, verbose=self.verbose)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError

MP2 = GMP2

scf.ghf.GHF.MP2 = lib.class_as_method(MP2)

def ao2mo_slow(eri_ao, mo_coeffs):
    nao = mo_coeffs[0].shape[0]
    eri_ao_s1 = ao2mo.restore(1, eri_ao, nao)
    return lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao_s1.reshape([nao]*4),
                          mo_coeffs[0].conj(), mo_coeffs[1],
                          mo_coeffs[2].conj(), mo_coeffs[3])

def _make_eris_incore(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    eris = gmp2._PhysicistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    orbspin = eris.orbspin

    if callable(ao2mofn):
        orbo = eris.mo_coeff[:,:nocc]
        orbv = eris.mo_coeff[:,nocc:]
        orbo = lib.tag_array(orbo, orbspin=orbspin)
        eri = ao2mofn((orbo,orbv,orbo,orbv)).reshape(nocc,nvir,nocc,nvir)
    else:
        orboa = eris.mo_coeff[:nao//2,:nocc]
        orbob = eris.mo_coeff[nao//2:,:nocc]
        orbva = eris.mo_coeff[:nao//2,nocc:]
        orbvb = eris.mo_coeff[nao//2:,nocc:]
        if orbspin is None:
            eri  = ao2mo_slow(mp._scf._eri, (orboa,orbva,orboa,orbva))
            eri += ao2mo_slow(mp._scf._eri, (orbob,orbvb,orbob,orbvb)) 
            eri1 = ao2mo_slow(mp._scf._eri, (orboa,orbva,orbob,orbvb))
            eri += eri1
            eri += eri1.transpose(2,3,0,1)
        else:
            co = orboa + orbob
            cv = orbva + orbvb
            eri = ao2mo_slow(mp._scf._eri, (co,cv,co,cv))
            sym_forbid = (orbspin[:nocc,None] != orbspin[nocc:])
            eri[sym_forbid,:,:] = 0
            eri[:,:,sym_forbid] = 0

    eris.oovv = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    return eris

del(WITH_T2)


if __name__ == '__main__':
    from functools import reduce
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    frozen = [0,1,2,3]
    pt = GMP2(mf, frozen=frozen)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    pt.max_memory = 1000
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    dm1 = pt.make_rdm1(t2)
    dm2 = pt.make_rdm2(t2)
    nao = mol.nao_nr()
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = mf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - pt.e_tot)

    mf = scf.UHF(mol).run(max_cycle=1)
    mf = scf.addons.convert_to_ghf(mf)
    pt = GMP2(mf)
    print(pt.kernel()[0] - -0.371240143556976)
