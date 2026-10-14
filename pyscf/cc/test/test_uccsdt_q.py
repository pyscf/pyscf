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

"""UHF quadruples corrections against independent determinant calculations."""

import copy
import importlib.util
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pyscf import cc, gto, scf
from pyscf.lib import logger
from pyscf.cc import uccsdt, uccsdt_q
from pyscf.cc.test import uccsdt_q_reference as reference
from pyscf.cc.test import uccsdt_q_slow as slow
from pyscf.cc.test.test_uccsdt_q_reference import random_inputs


def random_native(nalpha, compact=False):
    """Native blocks from an independent spin-orbital random fixture."""
    _, t2, t3, eps, _, bare = random_inputs(nalpha)
    a, b = np.arange(nalpha), np.arange(nalpha, 4)
    alpha, beta = np.r_[a, a + 4], np.r_[b, b + 4]
    mycc = SimpleNamespace(nocc=(len(a), len(b)), nmo=(2 * len(a), 2 * len(b)),
                           cc_order=3, do_tri_max_t=compact, max_memory=1000,
                           stdout=io.StringIO(), verbose=0, level_shift=0.,
                           einsum_backend='numpy', e_tot=-1.0, e_corr=-0.1,
                           converged=True, mo_coeff=(np.eye(2 * len(a)), np.eye(2 * len(b))))
    eris = SimpleNamespace(
        pppp=np.ascontiguousarray(bare[np.ix_(alpha, alpha, alpha, alpha)]),
        PPPP=np.ascontiguousarray(bare[np.ix_(beta, beta, beta, beta)]),
        pPpP=np.ascontiguousarray(bare[np.ix_(alpha, beta, alpha, beta)]),
        mo_energy=(eps[alpha].copy(), eps[beta].copy()), nocc=mycc.nocc,
        fock=(np.diag(eps[alpha]), np.diag(eps[beta])))
    t1_native = (np.zeros((len(a), len(a))), np.zeros((len(b), len(b))))
    t2_native = (t2[np.ix_(a, a, a, a)],
                 t2[np.ix_(a, b, a, b)].transpose(0, 2, 1, 3).copy(),
                 t2[np.ix_(b, b, b, b)])
    t3_native = (t3[np.ix_(a, a, a, a, a, a)],
                 t3[np.ix_(a, a, b, a, a, b)].transpose(0, 1, 3, 4, 2, 5).copy(),
                 t3[np.ix_(b, b, a, b, b, a)].transpose(0, 1, 3, 4, 2, 5).copy(),
                 t3[np.ix_(b, b, b, b, b, b)])
    if compact:
        t3_native = uccsdt.tamps_full2tri_uhf(mycc, t3_native)
    mycc.tamps = [t1_native, t2_native, t3_native]
    mycc.ao2mo = lambda *args: eris
    return mycc, eris


def setUpModule():
    global beh, bh2, beh_eris, bh2_eris
    mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    beh = cc.UCCSDT(mf, compact_tamps=True).run(conv_tol=1e-10)
    beh_eris = beh.ao2mo()
    mol = gto.M(atom='B 0 0 0; H -1.0392304845 0 0.6; H 1.0392304845 0 0.6',
                basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    bh2 = cc.UCCSDT(mf, frozen=1, compact_tamps=True).run(conv_tol=1e-10)
    bh2_eris = bh2.ao2mo()


class KnownValues(unittest.TestCase):
    def test_five_spin_blocks_full_compact(self):
        for nalpha in range(5):
            for compact in (False, True):
                with self.subTest(nalpha=nalpha, compact=compact):
                    mycc, eris = random_native(nalpha, compact)
                    expected = reference.kernel(mycc, eris)
                    actual = uccsdt_q.kernel(mycc, eris)
                    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_production_contraction_classes(self):
        # Every spin sector has four occupied/virtual spin orbitals, so the
        # full reference tensors stay small. Check each contraction before
        # summing energies to exclude cancellation between different terms.
        for nalpha in range(5):
            mycc, eris = random_native(nalpha)
            inputs = reference.spinorbital_inputs(mycc, eris)
            expanded = slow.terms(inputs['v'], inputs['t2'], inputs['t3'])
            spins = (0,) * nalpha + (1,) * (4 - nalpha)
            virtual = [np.arange(nalpha) if s == 0 else np.arange(nalpha, 4)
                       for s in spins]
            for compact in (False, True):
                mycc, eris = random_native(nalpha, compact)
                nocc = nvir = mycc.nocc
                amplitudes = uccsdt_q._Amplitudes(mycc.tamps, nocc, nvir, compact)
                integrals = uccsdt_q._IntegralBlocks(eris, nocc)
                ranges = {(x, s): np.arange(nocc[s]) for x in 'mnef' for s in (0, 1)}
                holes = tuple(range(nalpha)) + tuple(range(4 - nalpha))
                for x, s, i in zip('ijkl', spins, holes):
                    ranges[x, s] = np.array([i])
                for x, s in zip('abcd', spins):
                    ranges[x, s] = np.arange(nvir[s])
                shape = tuple(nvir[s] for s in spins)
                for name, tensor in expanded.items():
                    with self.subTest(nalpha=nalpha, compact=compact, contraction=name):
                        terms = [term for term in uccsdt_q._spin_terms(nalpha) if term[0] == name]
                        w4, l3, l2 = uccsdt_q._moments(
                            amplitudes, integrals, terms, ranges, shape,
                            lambda script, *arrays: np.einsum(script, *arrays, optimize=True))
                        expected = tensor[0, 1, 2, 3][np.ix_(*virtual)]
                        np.testing.assert_allclose(w4, 0. if name == 'l2' else expected,
                                                   atol=1e-11, rtol=1e-10)
                        np.testing.assert_allclose(l2, expected if name == 'l2' else 0.,
                                                   atol=1e-11, rtol=1e-10)
                        np.testing.assert_allclose(l3, expected if name in ('vvov', 'vooo') else 0.,
                                                   atol=1e-11, rtol=1e-10)

    def test_open_shell_known_values(self):
        # These values come from the bit-string determinant oracle, not an
        # external CC package.  Both geometries are fixed, neutral doublets.
        for mycc, eris, expected in (
                (beh, beh_eris, (2.5486782537324673e-8, -4.584336210071545e-8)),
                (bh2, bh2_eris, (-1.613660873008072e-6, -3.3604280109780666e-6))):
            with self.subTest(atom=mycc.mol.atom):
                self.assertTrue(mycc._scf.converged)
                self.assertTrue(mycc.converged)
                actual = mycc.ccsdt_q(eris=eris)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-8)
                np.testing.assert_allclose(actual, reference.kernel(mycc, eris), atol=1e-12, rtol=1e-10)
                highm = cc.UCCSDT(mycc._scf, frozen=mycc.frozen, compact_tamps=False)
                highm.tamps = [*mycc.tamps[:2], mycc.tamps_tri2full(mycc.tamps[2])]
                np.testing.assert_allclose(highm.ccsdt_q(eris=eris), actual, atol=1e-12, rtol=1e-10)

    def test_closed_shell_limit(self):
        mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        rhf = cc.RCCSDT(mf, compact_tamps=False).run(conv_tol=1e-10)
        uhf = cc.UCCSDT(scf.addons.convert_to_uhf(mf), compact_tamps=False)
        uhf.tamps = uccsdt.tamps_rhf2uhf(uhf, rhf.tamps)
        np.testing.assert_allclose(uhf.ccsdt_q(), rhf.ccsdt_q(), atol=1e-11, rtol=1e-8)

    def test_frozen_orbital_forms(self):
        # Occupied and virtual freezing also exercise unequal alpha/beta
        # dimensions. Nonzero compact T3 values exercise their unpacking;
        # initial T2 avoids five duplicate CC solutions.
        rng = np.random.default_rng(816)
        for frozen in (1, [0], [[0], [0]], [[0, 6], [0]], [[0], [0, 1]]):
            with self.subTest(frozen=frozen):
                mycc = cc.UCCSDT(bh2._scf, frozen=frozen, compact_tamps=True)
                eris = mycc.ao2mo()
                _, tamps = mycc.init_amps(eris)
                tamps[2] = [rng.normal(size=block.shape) * 1e-3 for block in tamps[2]]
                actual = mycc.ccsdt_q(tamps=tamps, eris=eris)
                expected = reference.kernel(mycc, eris, tamps)
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)

    def test_einsum_backends(self):
        backends = ['numpy', 'pyscf']
        if importlib.util.find_spec('pytblis') is not None:
            backends.append('pytblis')
        for nalpha, compact in ((1, False), (2, True), (3, True)):
            mycc, eris = random_native(nalpha, compact=compact)
            expected = reference.kernel(mycc, eris)
            np.testing.assert_allclose(uccsdt_q.kernel(mycc, eris), expected,
                                       atol=1e-11, rtol=1e-10)
            for backend in backends:
                with self.subTest(backend=backend, nalpha=nalpha, compact=compact):
                    mycc.einsum_backend = backend
                    np.testing.assert_allclose(uccsdt_q._kernel_python(mycc, eris), expected,
                                               atol=1e-11, rtol=1e-10)

    def test_alpha_beta_exchange(self):
        for compact in (False, True):
            mycc, eris = random_native(1, compact)
            expected = uccsdt_q.kernel(mycc, eris)
            flipped = copy.copy(mycc)
            flipped.nocc, flipped.nmo = mycc.nocc[::-1], mycc.nmo[::-1]
            t1, t2, t3 = mycc.tamps
            flipped.tamps = [t1[::-1], (t2[2], t2[1].transpose(2, 3, 0, 1), t2[0]), t3[::-1]]
            flipped_eris = SimpleNamespace(pppp=eris.PPPP, PPPP=eris.pppp,
                                          pPpP=eris.pPpP.transpose(1, 0, 3, 2),
                                          mo_energy=eris.mo_energy[::-1],
                                          fock=eris.fock[::-1], nocc=flipped.nocc)
            np.testing.assert_allclose(uccsdt_q.kernel(flipped, flipped_eris), expected,
                                       atol=1e-11, rtol=1e-10)

    def test_inputs_and_energy_attributes_unchanged(self):
        mycc, eris = random_native(2, True)
        before_tamps = copy.deepcopy(mycc.tamps)
        before_eris = copy.deepcopy(eris)
        expected = uccsdt_q.kernel(mycc, eris)
        self.assertEqual(mycc.e_tot, -1.0)
        self.assertEqual(mycc.e_corr, -0.1)
        for old, new in zip(before_tamps, mycc.tamps):
            for x, y in zip(old, new):
                np.testing.assert_array_equal(x, y)
        for key in ('pppp', 'PPPP', 'pPpP'):
            np.testing.assert_array_equal(getattr(eris, key), getattr(before_eris, key))
        for key in ('fock', 'mo_energy'):
            for x, y in zip(getattr(eris, key), getattr(before_eris, key)):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(uccsdt_q.kernel(mycc, tamps=before_tamps), expected,
                                   atol=1e-12, rtol=1e-10)

    def test_t1_and_level_shift_independence(self):
        mycc, eris = random_native(2, True)
        expected = uccsdt_q.kernel(mycc, eris)
        mycc.level_shift = 0.8
        altered = [tuple(x + 0.3 for x in mycc.tamps[0]), *mycc.tamps[1:]]
        np.testing.assert_allclose(uccsdt_q.kernel(mycc, eris, altered), expected,
                                   atol=1e-12, rtol=1e-10)

    def test_noncanonical_and_invalid_inputs(self):
        mycc, eris = random_native(2)
        noncanonical = copy.deepcopy(eris)
        noncanonical.fock[0][0, 1] = noncanonical.fock[0][1, 0] = 1e-4
        with self.assertRaises((ValueError, NotImplementedError)):
            uccsdt_q.kernel(mycc, noncanonical)
        for dtype in (np.complex128, np.float32):
            bad_eris = copy.deepcopy(eris)
            bad_eris.pPpP = bad_eris.pPpP.astype(dtype)
            with self.subTest(dtype=dtype):
                with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                    uccsdt_q.kernel(mycc, bad_eris)
        malformed = list(mycc.tamps)
        malformed[1] = (mycc.tamps[1][0][:-1], *mycc.tamps[1][1:])
        with self.assertRaises((ValueError, TypeError)):
            uccsdt_q.kernel(mycc, eris, malformed)
        with self.assertRaises((ValueError, TypeError)):
            uccsdt_q.kernel(mycc, eris, mycc.tamps[:2])

    def test_invalid_orbital_energies_and_fock(self):
        # A NaN denominator must not be mistaken for a zero denominator and
        # silently turn a nonzero correction into (0, 0).
        mycc, eris = random_native(2)
        for name in ('mo_energy', 'fock'):
            for spin in (0, 1):
                for value in (np.nan, np.inf):
                    with self.subTest(array=name, spin=spin, value=value):
                        invalid = copy.deepcopy(eris)
                        array = getattr(invalid, name)[spin]
                        array.flat[0] = value
                        with self.assertRaises(ValueError):
                            uccsdt_q.kernel(mycc, invalid)
        for spin in (0, 1):
            with self.subTest(energy_shape_spin=spin):
                invalid = copy.deepcopy(eris)
                invalid.mo_energy = tuple(e[:-1] if s == spin else e
                                          for s, e in enumerate(eris.mo_energy))
                with self.assertRaises(ValueError):
                    uccsdt_q.kernel(mycc, invalid)

    def test_zero_denominators(self):
        mycc, eris = random_native(2)
        eris.mo_energy = tuple(np.zeros_like(e) for e in eris.mo_energy)
        eris.fock = tuple(np.diag(e) for e in eris.mo_energy)
        actual = uccsdt_q.kernel(mycc, eris, verbose=logger.WARN)
        np.testing.assert_array_equal(actual, (0., 0.))
        self.assertIn('denominator', mycc.stdout.getvalue().lower())

    def test_no_quadruple_excitations(self):
        mycc = cc.UCCSDT(beh._scf, frozen=1, compact_tamps=True)
        eris = mycc.ao2mo()
        _, tamps = mycc.init_amps(eris)
        np.testing.assert_array_equal(mycc.ccsdt_q(tamps=tamps, eris=eris), (0., 0.))

    def test_virtual_blocking(self):
        expected = reference.kernel(beh, beh_eris)
        with mock.patch.object(uccsdt_q, '_native_block_size', return_value=1):
            actual = beh.ccsdt_q(eris=beh_eris)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


if __name__ == '__main__':
    print('Full Tests for UCCSDT(Q)')
    unittest.main()
