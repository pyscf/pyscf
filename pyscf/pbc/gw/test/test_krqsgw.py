#!/usr/bin/env python

import numpy as np
import pytest

from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.gw import krqsgw, kuqsgw
from pyscf.pbc.gw.krqsgw import KRQSGW
from pyscf.pbc.gw.test.test_krgw_ac import h2_hf, run_small_gw


@pytest.fixture(scope="module")
def diamond_pbe():
    cell = gto.Cell()
    cell.build(
        unit="angstrom",
        a="""
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        """,
        atom="C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375",
        dimension=3,
        verbose=0,
        output="/dev/null",
        pseudo="gth-pade",
        basis="gth-szv",
        precision=1e-10,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf.build()

    kmf = dft.KRKS(cell, kpts).rs_density_fit()
    kmf.xc = "pbe"
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_krqsgw_no_fc_high_cost(diamond_pbe):
    gw = KRQSGW(diamond_pbe)
    gw.fc = False
    gw.kernel()

    assert gw.mo_energy[0][3] == pytest.approx(0.71535002, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.99042025, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.53587079, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.14822789, abs=1e-4)


@pytest.fixture(scope="module")
def heh_hf():
    cell = gto.M(atom='He 0 0 0; H 0 0 1.', charge=1, a=np.eye(3)*4,
                 basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                 precision=1e-7, verbose=0)
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
    mf = mf.density_fit(auxbasis='weigend').run(conv_tol=1e-11)
    return mf


def check_mode(mf, mode, finite_size):
    ref = krqsgw.KRQSGW(mf)
    spin = kuqsgw.KUQSGW(mf.to_uhf())
    for obj in (ref, spin):
        obj.mode = mode
        obj.fc = finite_size
        obj.nw = 32
        obj.ac_pade_npts = 8
        obj.max_cycle = 60
        obj.conv_tol = 1e-8
        obj.kernel()
        assert obj.converged
    np.testing.assert_allclose(spin.mo_energy[0], ref.mo_energy, atol=2e-7)
    np.testing.assert_allclose(spin.mo_energy[1], ref.mo_energy, atol=2e-7)
    np.testing.assert_allclose(spin.vsig[0], ref.vsig, atol=2e-7)


def test_mode_a_finite_size(heh_hf):
    check_mode(heh_hf, 'a', True)


def test_mode_c(heh_hf):
    check_mode(heh_hf, 'c', False)


def test_quasiparticle_self_consistency_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krqsgw.KRQSGW, max_cycle=100, conv_tol=1e-8)
    out = run_small_gw(h2_hf, kuqsgw.KUQSGW, unrestricted=True, max_cycle=100, conv_tol=1e-8)
    assert ref.converged
    assert out.converged
    np.testing.assert_allclose(out.mo_energy[0], ref.mo_energy, atol=2e-7)
    np.testing.assert_allclose(out.mo_energy[1], ref.mo_energy, atol=2e-7)
