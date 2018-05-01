#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.scf import _vhf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = '631g',
)

mf = scf.RHF(mol)
mf.scf()
nao, nmo = mf.mo_coeff.shape

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnowValues(unittest.TestCase):
    def test_incore_s4(self):
        eri4 = ao2mo.restore(4, mf._eri, nmo)
        dm = mf.make_rdm1()
        vj0, vk0 = _vhf.incore(eri4, dm, hermi=1)
        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_mapdm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        eri0 = mol.intor('int2e_ip1_sph', comp=3).reshape(3,nao,nao,nao,nao)
        vj0 = numpy.einsum('nijkl,lk->nij', eri0, dm)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_bindm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        vj0, vk0 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        dms = (dm,dm)
        vj1, vk1 = _vhf.direct_bindm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dms, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_rdirect_mapdm(self):
        numpy.random.seed(1)
        n2c = nao*2
        dm = numpy.random.random((n2c,n2c)) + \
             numpy.random.random((n2c,n2c)) * 1j
        eri0 = mol.intor('int2e_g1_spinor', comp=3).reshape(3,n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.rdirect_mapdm('int2e_g1_spinor', 'a4ij',
                                      ('lk->s2ij', 'jk->s1il'),
                                      dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_rdirect_bindm_high_cost(self):
        n2c = nao*2
        mfr = scf.DHF(mol)
        mfr.scf()
        dm = mfr.make_rdm1()[:n2c,:n2c].copy()

        eri0 = mol.intor('int2e_spsp1_spinor').reshape(n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('int2e_spsp1_spinor', 's4', ('jk->s1il',),
                                 (dm,), 1, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))


if __name__ == "__main__":
    print("Full Tests for _vhf")
    unittest.main()


