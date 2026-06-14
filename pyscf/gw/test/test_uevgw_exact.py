#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.uevgw_exact import UEVGWExact


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


def test_uevgw_exact(h2o_cation_pbe0):
    gw = UEVGWExact(h2o_cation_pbe0)
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.04866247, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15108235, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.00823012, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.40657385, abs=1e-4)


def test_uevgw_exact_w0(h2o_cation_pbe0):
    gw = UEVGWExact(h2o_cation_pbe0)
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.03700493, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15367477, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99996820, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.41598473, abs=1e-4)
