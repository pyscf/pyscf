#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.qsgw_exact import QSGWExact


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


def test_qsgw_exact(hf_pbe0):
    gw = QSGWExact(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38783292, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439482, abs=1e-4)
