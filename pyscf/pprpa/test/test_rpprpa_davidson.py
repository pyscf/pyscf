#!/usr/bin/env python

import pytest
from pyscf import gto, dft
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson


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

# ppRPA
def test_pprpa_singlet_triplet(h2o_dication_rks):
    pp = RppRPADavidson(h2o_dication_rks, nvir_act=10, nroot=3)
    pp.kernel('s')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_s == pytest.approx([0.92727944, 1.18456136, 1.25715794], abs=1e-5)
    assert pp.exci_t == pytest.approx([1.16289368, 1.24603019, 1.64840497], abs=1e-5)

def test_pprpa_singlet(h2o_dication_rks):
    pp = RppRPADavidson(h2o_dication_rks, nvir_act=10, nroot=3)
    pp.kernel('s')
    pp.analyze()
    assert pp.exci_s == pytest.approx([0.92727944, 1.18456136, 1.25715794], abs=1e-5)

def test_pprpa_triplet(h2o_dication_rks):
    pp = RppRPADavidson(h2o_dication_rks, nvir_act=10, nroot=3)
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_t == pytest.approx([1.16289368, 1.24603019, 1.64840497], abs=1e-5)

# hhRPA
def test_hhrpa_singlet_triplet(h2o_dianion_rks):
    pp = RppRPADavidson(h2o_dianion_rks, nvir_act=10, nroot=3, channel='hh')
    pp.kernel('s')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_s == pytest.approx([-0.39661544, -0.63812611, -0.72406249], abs=1e-5)
    assert pp.exci_t == pytest.approx([-0.62371585, -0.69696579, -0.83182685], abs=1e-5)

def test_hhrpa_singlet(h2o_dianion_rks):
    pp = RppRPADavidson(h2o_dianion_rks, nvir_act=10, nroot=3, channel='hh')
    pp.kernel('s')
    pp.analyze()
    assert pp.exci_s == pytest.approx([-0.39661544, -0.63812611, -0.72406249], abs=1e-5)

def test_hhrpa_triplet(h2o_dianion_rks):
    pp = RppRPADavidson(h2o_dianion_rks, nvir_act=10, nroot=3, channel='hh')
    pp.kernel('t')
    pp.analyze()
    assert pp.exci_t == pytest.approx([-0.62371585, -0.69696579, -0.83182685], abs=1e-5)
