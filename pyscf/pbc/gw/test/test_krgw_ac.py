#!/usr/bin/env python

import pytest

from pyscf.pbc import df, dft, gto
from pyscf.pbc.gw.krgw_ac import KRGWAC


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


def test_krgwac_pade_no_fc(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = False
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.62044205, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.96572609, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.52637438, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.07517363, abs=1e-4)


def test_krgwac_pade_no_fc_outcore(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = False
    gw.outcore = True
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.62044205, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.96572609, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.52637438, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.07517363, abs=1e-4)


def test_krgwac_pade_with_fc(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = True
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.44025061, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.80148565, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.35193483, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(0.92909525, abs=1e-4)


def test_krgwac_pade_with_fc_frozen_core(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = True
    gw.frozen = 1
    gw.kernel()

    assert gw.mo_energy[0][3] == pytest.approx(0.44092615, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.79820946, abs=1e-4)
