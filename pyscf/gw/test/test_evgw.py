#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.evgw import EVGW


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


def test_evgw(h2o_pbe0):
    gw = EVGW(h2o_pbe0)
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.44302542, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.1681912, abs=1e-4)


def test_evgw0(h2o_pbe0):
    gw = EVGW(h2o_pbe0)
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.43324833, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16626621, abs=1e-4)
