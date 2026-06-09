#!/usr/bin/env python

import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.kuevgw import KUEVGW


@pytest.fixture(scope="module")
def hydrogen_kuhf():
    cell = gto.Cell()
    cell.build(
        unit="B",
        a=[[0.0, 6.74027466, 6.74027466], [6.74027466, 0.0, 6.74027466], [6.74027466, 6.74027466, 0.0]],
        atom="""H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733""",
        basis="gth-dzvp",
        pseudo="gth-pade",
        verbose=0,
        output="/dev/null",
        charge=0,
        spin=3,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf.build()

    kmf = scf.KUHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_kuevgw_with_fc(hydrogen_kuhf):
    gw = KUEVGW(hydrogen_kuhf)
    gw.fc = True
    gw.kernel()

    assert gw.mo_energy[0][0][1] == pytest.approx(-0.28173406, abs=5e-4)
    assert gw.mo_energy[0][0][2] == pytest.approx(0.13301134, abs=5e-4)
    assert gw.mo_energy[1][1][0] == pytest.approx(-0.33469053, abs=5e-4)
    assert gw.mo_energy[1][1][1] == pytest.approx(0.07654302, abs=5e-4)
