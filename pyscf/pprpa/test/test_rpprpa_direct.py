#!/usr/bin/env python

import pytest
from pyscf import gto, dft
from pyscf.pprpa.rpprpa_direct import RppRPADirect


def h2o_b3lyp(charge):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.charge = charge
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    return mf

@pytest.fixture(scope="module")
def h2o_dication_rks():
    # (N-2)-electron reference for ppRPA
    mf = h2o_b3lyp(charge=2)
    yield mf
    mf.mol.stdout.close()

@pytest.fixture(scope="module")
def h2o_dianion_rks():
    # (N+2)-electron reference for hhRPA
    mf = h2o_b3lyp(charge=-2)
    yield mf
    mf.mol.stdout.close()

@pytest.fixture(scope="module")
def h2o_rks():
    mf = h2o_b3lyp(charge=0)
    yield mf
    mf.mol.stdout.close()

# ppRPA
def test_pprpa_singlet_triplet(h2o_dication_rks):
    pp = RppRPADirect(h2o_dication_rks, nvir_act=10)
    pp.kernel('s')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_s[10:13] == pytest.approx([0.92727944, 1.18456136, 1.25715794], abs=1e-5)
    assert pp.exci_t[6:9] == pytest.approx([1.16289368, 1.24603019, 1.64840497], abs=1e-5)

def test_pprpa_singlet(h2o_dication_rks):
    pp = RppRPADirect(h2o_dication_rks, nvir_act=10)
    pp.kernel('s')
    pp.analyze()
    assert pp.exci_s[10:13] == pytest.approx([0.92727944, 1.18456136, 1.25715794], abs=1e-5)

def test_pprpa_triplet(h2o_dication_rks):
    pp = RppRPADirect(h2o_dication_rks, nvir_act=10)
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_t[6:9] == pytest.approx([1.16289368, 1.24603019, 1.64840497], abs=1e-5)

# hhRPA
def test_hhrpa_singlet_triplet(h2o_dianion_rks):
    pp = RppRPADirect(h2o_dianion_rks, nvir_act=10, nelec='n+2')
    pp.kernel('s')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_s[18:21] == pytest.approx([-0.72406249, -0.63812611, -0.39661544], abs=1e-5)
    assert pp.exci_t[12:15] == pytest.approx([-0.83182685, -0.69696579, -0.62371585], abs=1e-5)

def test_hhrpa_singlet(h2o_dianion_rks):
    pp = RppRPADirect(h2o_dianion_rks, nvir_act=10, nelec='n+2')
    pp.kernel('s')
    pp.analyze()
    assert pp.exci_s[18:21] == pytest.approx([-0.72406249, -0.63812611, -0.39661544], abs=1e-5)

def test_hhrpa_triplet(h2o_dianion_rks):
    pp = RppRPADirect(h2o_dianion_rks, nvir_act=10, nelec='n+2')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_t[12:15] == pytest.approx([-0.83182685, -0.69696579, -0.62371585], abs=1e-5)

# correlation energy
def test_correlation_singlet_triplet(h2o_rks):
    pp = RppRPADirect(h2o_rks)
    etot, _, ec = pp.energy_tot()
    assert ec == pytest.approx(-0.1873322028, abs=1e-7)
    assert etot == pytest.approx(-76.1447465523, abs=1e-7)

def test_correlation_singlet(h2o_rks):
    pp = RppRPADirect(h2o_rks)
    pp.kernel('s')
    assert pp.ec_s == pytest.approx(-0.1126062628, abs=1e-7)

def test_correlation_triplet(h2o_rks):
    pp = RppRPADirect(h2o_rks)
    pp.kernel('t')
    assert pp.ec_t == pytest.approx(-0.07472594, abs=1e-7)
