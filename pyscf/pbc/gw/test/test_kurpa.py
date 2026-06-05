#!/usr/bin/env python

import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.kurpa import KURPA


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

    kmf = scf.KUHF(cell, kpts, exxdiv="ewald")
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_kurpa_no_fc(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = False
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04288352903004621, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-1.584806462873674, abs=1e-6)


def test_kurpa_no_fc_outcore(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = False
    rpa.outcore = True
    rpa.segsize = 3
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04288352903004621, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-1.584806462873674, abs=1e-6)


def test_kurpa_with_fc(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04295466718074476, abs=1e-6)
