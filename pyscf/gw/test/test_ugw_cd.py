#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.ugw_cd import UGWCD


@pytest.fixture
def h2o_cation_uks_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_ugw_cd(h2o_cation_uks_pbe0):
    gw = UGWCD(h2o_cation_uks_pbe0)
    gw.orbs = range(0, 8)
    gw.kernel()

    assert gw.mo_energy[0][0] == pytest.approx(-20.32245841, abs=1e-5)
    assert gw.mo_energy[1][0] == pytest.approx(-20.28545214, abs=1e-5)
