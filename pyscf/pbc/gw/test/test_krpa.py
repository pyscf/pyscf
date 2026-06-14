#!/usr/bin/env python

import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.krpa import KRPA


@pytest.fixture(scope="module")
def diamond_krhf():
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
        pseudo="gth-pbe",
        basis="gth-dzv",
        precision=1e-12,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSGDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_krpa_no_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = False
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.1852772037535004, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.694392044197565, abs=1e-6)


def test_krpa_no_fc_outcore(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.outcore = True
    rpa.segsize = 2
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.1852772037535004, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.694392044197565, abs=1e-6)


def test_krpa_with_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.20723389722097715, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.716348738655793, abs=1e-6)
