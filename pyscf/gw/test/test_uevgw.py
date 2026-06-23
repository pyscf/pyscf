#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.uevgw import UEVGW


@pytest.fixture
def h2o_cation_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_uevgw(h2o_cation_pbe0):
    gw = UEVGW(h2o_cation_pbe0)
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.04870918, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15114275, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.00827479, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.40647955, abs=1e-4)


def test_uevgw0(h2o_cation_pbe0):
    gw = UEVGW(h2o_cation_pbe0)
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.03696692, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15367283, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99997861, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.41597008, abs=1e-4)
