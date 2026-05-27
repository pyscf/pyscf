#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy

from pyscf import gto, scf

try:
    from pyscf.tools import trexio as ptr
    import trexio as _trexio  # noqa: F401
    HAVE_TREXIO = True
except ImportError:
    HAVE_TREXIO = False

# The HDF5 backend may be disabled when trexio is built without HDF5.
# Probe at module load and pick whichever backend works.
_BACKEND = 'TEXT'
if HAVE_TREXIO:
    with tempfile.TemporaryDirectory() as _d:
        try:
            _f = _trexio.File(os.path.join(_d, 'probe.h5'),
                              mode='w', back_end=_trexio.TREXIO_HDF5)
            _f.close()
            _BACKEND = 'HDF5'
        except _trexio.Error:
            _BACKEND = 'TEXT'


@unittest.skipUnless(HAVE_TREXIO, 'trexio not installed')
class KnownValues(unittest.TestCase):
    def _roundtrip_mf(self, mf, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'mf.trexio')
            ptr.to_trexio(mf, fn, backend=_BACKEND, **kwargs)
            mf2 = ptr.scf_from_trexio(fn, backend=_BACKEND)
            return mf2, fn

    def test_rhf_spherical(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', verbose=0)
        mf = scf.RHF(mol).run()
        mf2, _ = self._roundtrip_mf(mf)
        self.assertAlmostEqual(mf.e_tot, mf2.e_tot, 10)
        self.assertAlmostEqual(
            abs(mol.intor('int1e_ovlp')
                - mf2.mol.intor('int1e_ovlp')).max(), 0.0, 10)

    def test_rhf_cartesian_higher_l(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24',
                    basis='ccpvdz', cart=True, verbose=0)
        mf = scf.RHF(mol).run()
        mf2, _ = self._roundtrip_mf(mf)
        self.assertAlmostEqual(mf.e_tot, mf2.e_tot, 9)

    def test_uhf_open_shell(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='631g',
                    spin=2, verbose=0)
        mf = scf.UHF(mol).run()
        mf2, _ = self._roundtrip_mf(mf)
        self.assertIsInstance(mf2, scf.uhf.UHF)
        self.assertAlmostEqual(mf.e_tot, mf2.e_tot, 9)
        self.assertEqual(mf2.mo_coeff.shape, mf.mo_coeff.shape)

    def test_ecp_round_trip(self):
        mol = gto.M(atom='Cu 0 0 0; H 0 0 1.5',
                    basis={'Cu': 'lanl2dz', 'H': 'sto3g'},
                    ecp={'Cu': 'lanl2dz'}, verbose=0)
        mf = scf.RHF(mol).run()
        mf2, _ = self._roundtrip_mf(mf)
        self.assertAlmostEqual(mf.e_tot, mf2.e_tot, 9)
        self.assertTrue(mf2.mol.has_ecp())

    def test_ao_integrals(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='631g', verbose=0)
        mf = scf.RHF(mol).run()
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'mf.trexio')
            ptr.to_trexio(mf, fn, backend=_BACKEND,
                          with_ao_ints=True, with_eri=True)
            mf2 = ptr.scf_from_trexio(fn, backend=_BACKEND)
            ints = ptr.read_ao_1e_integrals(fn, mol=mf2.mol, backend=_BACKEND)
            eri = ptr.read_ao_2e_integrals(fn, mol=mf2.mol, backend=_BACKEND)
        self.assertAlmostEqual(
            abs(ints['overlap'] - mol.intor('int1e_ovlp')).max(), 0, 12)
        self.assertAlmostEqual(
            abs(ints['kinetic'] - mol.intor('int1e_kin')).max(), 0, 12)
        self.assertAlmostEqual(
            abs(ints['hcore'] - mf.get_hcore()).max(), 0, 12)
        self.assertAlmostEqual(
            abs(eri - mol.intor('int2e')).max(), 0, 12)

    def test_normalize_gto_restored_on_error(self):
        import pyscf.gto.mole as molmod
        saved = molmod.NORMALIZE_GTO
        try:
            ptr.mol_from_trexio('/nonexistent/path/file.trexio',
                                backend=_BACKEND)
        except Exception:
            pass
        self.assertEqual(molmod.NORMALIZE_GTO, saved)

    def test_canonical_cartesian_in_file(self):
        # The file should represent unit-normalised Cartesian AOs.
        import trexio as _t
        mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24',
                    basis='ccpvdz', cart=True, verbose=0)
        mf = scf.RHF(mol).run()
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'h2o.trexio')
            ptr.to_trexio(mf, fn, backend=_BACKEND, with_ao_ints=True)
            be = _t.TREXIO_HDF5 if _BACKEND == 'HDF5' else _t.TREXIO_TEXT
            tf = _t.File(fn, mode='r', back_end=be)
            S_file = numpy.asarray(_t.read_ao_1e_int_overlap(tf))
            tf.close()
        self.assertAlmostEqual(abs(S_file.diagonal() - 1).max(), 0, 12)

    def test_mo_eri_round_trip(self):
        from pyscf import ao2mo
        mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='631g', verbose=0)
        mf = scf.RHF(mol).run()
        nmo = mf.mo_coeff.shape[1]
        ref = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff, compact=False), nmo)
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'hf.trexio')
            ptr.to_trexio(mf, fn, backend=_BACKEND, with_mo_eri=True)
            mo_eri = ptr.read_mo_2e_integrals(fn, backend=_BACKEND)
        self.assertEqual(mo_eri.shape, (nmo, nmo, nmo, nmo))
        self.assertAlmostEqual(abs(mo_eri - ref).max(), 0, 12)

    def test_mol_only(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='631g**', verbose=0)
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'mol.trexio')
            ptr.to_trexio(mol, fn, backend=_BACKEND)
            mol2 = ptr.mol_from_trexio(fn, backend=_BACKEND)
        self.assertEqual(mol.nao_nr(), mol2.nao_nr())
        self.assertAlmostEqual(
            abs(mol.intor('int1e_ovlp')
                - mol2.intor('int1e_ovlp')).max(), 0, 12)


if __name__ == '__main__':
    print('Full tests for pyscf.tools.trexio')
    unittest.main()
