# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

"""Empty spin sectors must not allocate six-index residual workspaces."""

import copy
import unittest
from unittest import mock

import numpy as np

from pyscf import cc, gto, lib, scf
from pyscf.cc import uccsdt


class KnownValues(unittest.TestCase):
    def test_empty_sectors_and_full_update(self):
        cases = [('H 0 0 0; H 0 0 0.74', 'cc-pvtz', 0),
                 ('H 0 0 0; H 0 0 1; H 0 0 2.3', '6-31g', 1)]
        old_threads = lib.num_threads()
        lib.num_threads(1)
        try:
            for atom, basis, spin in cases:
                with self.subTest(basis=basis, spin=spin):
                    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
                    mf = (scf.RHF(mol).run().to_uhf() if spin == 0
                          else scf.UHF(mol).run())
                    compact = cc.UCCSDT(mf, compact_tamps=True)
                    eris = compact.ao2mo()
                    maps = compact.setup_tri2block()
                    names = ('t2c_map_6f_oa', 't2c_mask_6f_oa', 't2c_map_2f_oa', 't2c_mask_2f_oa',
                             't2c_map_6f_va', 't2c_mask_6f_va', 't2c_map_2f_va', 't2c_mask_2f_va',
                             't2c_map_6f_ob', 't2c_mask_6f_ob', 't2c_map_2f_ob', 't2c_mask_2f_ob',
                             't2c_map_6f_vb', 't2c_mask_6f_vb', 't2c_map_2f_vb', 't2c_mask_2f_vb')
                    for name, value in zip(names, maps):
                        setattr(compact, name, value)
                    nocc = max(compact.nocc)
                    nvir = max(m - o for m, o in zip(compact.nmo, compact.nocc))
                    compact.blksize_o_aaa = min(compact.blksize_o_aaa, nocc)
                    compact.blksize_o_aab = min(compact.blksize_o_aab, nocc)
                    compact.blksize_v_aaa = min(compact.blksize_v_aaa, nvir)
                    compact.blksize_v_aab = min(compact.blksize_v_aab, nvir)
                    compact.unique_tamps_map = compact.build_unique_tamps_map()
                    _, tamps = compact.init_amps(eris)
                    tamps = copy.deepcopy(tamps)

                    residuals = [uccsdt.compute_r3aaa_tri_uhf,
                                 uccsdt.compute_r3aab_tri_uhf,
                                 uccsdt.compute_r3bba_tri_uhf,
                                 uccsdt.compute_r3bbb_tri_uhf]
                    for index, residual in enumerate(residuals):
                        if tamps[2][index].size == 0:
                            with mock.patch.object(uccsdt.np, 'empty',
                                                   side_effect=AssertionError('workspace allocation')):
                                result = residual(compact, uccsdt._IMDS(),
                                                  tamps[1], tamps[2])
                            self.assertEqual(result.shape, tamps[2][index].shape)
                    if not any(x.size for x in tamps[2]):
                        with mock.patch.object(uccsdt.np, 'empty',
                                               side_effect=AssertionError('workspace allocation')):
                            uccsdt.r1r2_add_t3_tri_uhf_(
                                compact, uccsdt._IMDS(), None, None, tamps[2])
                    # Nonzero allowed triples exercise their feedback while
                    # the other spin sectors remain rigorously empty.
                    rng = np.random.default_rng(17)
                    for amplitude in tamps[2]:
                        amplitude[:] = rng.normal(scale=1e-3, size=amplitude.shape)
                    full = cc.UCCSDT(mf, compact_tamps=False)
                    full_tamps = copy.deepcopy(tamps[:2])
                    full_tamps.append(list(compact.tamps_tri2full(tamps[2])))
                    compact.update_amps_(tamps, eris)
                    full.update_amps_(full_tamps, eris)
                    actual = tamps[:2] + [compact.tamps_tri2full(tamps[2])]
                    for arank, brank in zip(actual, full_tamps):
                        for a, b in zip(arank, brank):
                            np.testing.assert_allclose(a, b, atol=3e-13, rtol=3e-13)
                    self.assertAlmostEqual(compact.energy(tamps, eris),
                                           full.energy(full_tamps, eris), 12)
        finally:
            lib.num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
