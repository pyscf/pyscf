#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.uevgw_exact import UEVGWExact


@pytest.fixture
def hf_cation_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.basis = 'sto-3g'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_uevgw_exact(hf_cation_pbe0):
    gw = UEVGWExact(hf_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.08983016, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.10536426, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.04970797, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.36252505, abs=1e-4)


def test_uevgw_exact_w0(hf_cation_pbe0):
    gw = UEVGWExact(hf_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.08983016, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.10536426, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.04970797, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.36252505, abs=1e-4)
