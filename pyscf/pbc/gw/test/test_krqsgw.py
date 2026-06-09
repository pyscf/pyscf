#!/usr/bin/env python

import pytest

from pyscf.pbc import df, dft, gto
from pyscf.pbc.gw.krqsgw import KRQSGW


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


def test_krqsgw_no_fc(diamond_pbe):
    gw = KRQSGW(diamond_pbe)
    gw.fc = False
    gw.kernel()

    assert gw.mo_energy[0][3] == pytest.approx(0.71535002, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.99042025, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.53587079, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.14822789, abs=1e-4)
