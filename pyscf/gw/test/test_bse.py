#!/usr/bin/env python

import pytest
from pyscf import dft, gto
from pyscf.gw.bse import BSE
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.ugw_ac import UGWAC


@pytest.fixture(scope='module')
def h2o_pbe_gw():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    gw = GWAC(mf)
    gw.kernel()
    return gw


@pytest.fixture(scope='module')
def h2o_cation_pbe_ugw():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    gw = UGWAC(mf)
    gw.kernel()
    return gw


def test_bse_singlet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('s')[0]
    assert exci[0] == pytest.approx(0.25749397, abs=1e-5)


def test_bse_triplet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('t')[0]
    assert exci[0] == pytest.approx(0.22299263, abs=1e-5)


def test_bse_unrestricted(h2o_cation_pbe_ugw):
    bse = BSE(h2o_cation_pbe_ugw)
    exci = bse.kernel('u')[0]
    assert exci[0] == pytest.approx(0.02114003, abs=1e-5)


def test_bse_energy_specific_singlet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('s', e_min=0.4)[0]
    assert exci[0] == pytest.approx(0.42691789, abs=1e-5)


def test_bse_energy_specific_triplet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('t', e_min=0.4)[0]
    assert exci[0] == pytest.approx(0.45195324, abs=1e-5)
