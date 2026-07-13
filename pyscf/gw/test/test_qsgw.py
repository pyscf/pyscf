#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.qsgw import QSGW


@pytest.fixture
def hf_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_qsgw(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439850, abs=1e-4)


def test_qsgw_low_nw2(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.nw2 = 30
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439851, abs=1e-4)


def test_qsgw_outcore(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.outcore = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439852, abs=1e-4)
