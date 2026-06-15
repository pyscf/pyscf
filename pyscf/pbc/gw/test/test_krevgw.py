#!/usr/bin/env python

import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.krevgw import KREVGW


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
        pseudo="gth-pade",
        basis="gth-szv",
        precision=1e-10,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_krevgw_no_fc_high_cost(diamond_krhf):
    gw = KREVGW(diamond_krhf)
    gw.fc = False
    gw.max_cycle = 100
    gw.kernel()

    assert gw.mo_energy[0][3] == pytest.approx(0.71068048, abs=5e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.99685827, abs=5e-4)
