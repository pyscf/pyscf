#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.qsgw import QSGW


@pytest.fixture
def h2o_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_qsgw(h2o_pbe0):
    gw = QSGW(h2o_pbe0)
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.45770292, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16254796, abs=1e-4)


def test_qsgw_low_nw2(h2o_pbe0):
    gw = QSGW(h2o_pbe0)
    gw.nw2 = 30
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.45770292, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16254796, abs=1e-4)


def test_qsgw_outcore(h2o_pbe0):
    gw = QSGW(h2o_pbe0)
    gw.outcore = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.45770292, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16254796, abs=1e-4)
