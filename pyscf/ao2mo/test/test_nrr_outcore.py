#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.ao2mo import nrr_outcore

class KnownValues(unittest.TestCase):
    def test_nrr_ghf(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            O     0.   0.       0.
            H     0.   -0.757   0.587
            H     0.   0.757    0.587'''
        mol.basis = '631g'
        mol.build()
        mf = mol.GHF().run()

        nao = mol.nao
        nmo = mf.mo_coeff.shape[1]
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        eri0 = mol.intor('int2e_sph')
        eri1  = lib.einsum('pqrs,pi,qj->ijrs', eri0, mo_a.conj(), mo_a)
        eri1 += lib.einsum('pqrs,pi,qj->ijrs', eri0, mo_b.conj(), mo_b)
        ref  = lib.einsum('ijrs,rk,sl->ijkl', eri1, mo_a.conj(), mo_a)
        ref += lib.einsum('ijrs,rk,sl->ijkl', eri1, mo_b.conj(), mo_b)

        eri1 = nrr_outcore.full_iofree(mol, mf.mo_coeff, 'h2oeri.h5')
        eri1 = eri1.reshape([nmo]*4)
        self.assertAlmostEqual(abs(ref - eri1).max(), 0, 11)

    def test_nrr_spinor(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            O     0.   0.       0.
            H     0.   -0.757   0.587
            H     0.   0.757    0.587'''
        mol.basis = '631g'
        mol.build()
        mf = mol.GHF().x2c().run()
        mo = mf.mo_coeff

        nao = mol.nao
        nmo = mf.mo_coeff.shape[1]
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        eri0 = mol.intor('int2e_spinor')
        ref = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri0, mo.conj(), mo, mo.conj(), mo)
        eri1 = nrr_outcore.full_iofree(mol, mo, 'h2oeri.h5', motype='j-spinor')
        eri1 = eri1.reshape([nmo]*4)
        self.assertAlmostEqual(abs(ref - eri1).max(), 0, 11)

    def test_component_integrals(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
        rng = np.random.default_rng(9)
        mo = rng.normal(size=(2*mol.nao, 3))+1j*rng.normal(size=(2*mol.nao, 3))
        spin = mo.reshape(2, mol.nao, 3)
        pair = np.einsum('spi,sqj->pqij', spin.conj(), spin)
        for intor, symmetry in (('int2e', 's1'), ('int2e_ip1', 's1')):
            with self.subTest(integral=intor):
                ao = mol.intor(intor)
                expected = lib.einsum('...pqrs,pqij,rskl->...ijkl', ao, pair, pair)
                actual = nrr_outcore.general_iofree(mol, (mo,)*4, intor=intor, aosym=symmetry)
                np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=1e-10)
        with lib.H5TmpFile() as archive:
            archive['eri_mo'] = np.zeros((1, 1))
            nrr_outcore.full(mol, mo, archive.filename, aosym='s1')
            expected = lib.einsum('pqrs,pqij,rskl->ijkl', mol.intor('int2e'), pair, pair)
            np.testing.assert_allclose(archive['eri_mo'][:].reshape(expected.shape), expected, atol=1e-10)


if __name__ == '__main__':
    print('Full Tests for ao2mo.nrr_outcore')
    unittest.main()
