#!/usr/bin/env python

import pytest
from pyscf import gto, dft
from pyscf.pprpa.upprpa_direct import UppRPADirect


def h2o_b3lyp(charge, spin):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.charge = charge
    mol.spin = spin
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    return mf

@pytest.fixture(scope="module")
def h2o_cation_uks():
    # spin-polarized (N-2)-electron reference for ppRPA, 5 alpha and 4 beta electrons
    mf = h2o_b3lyp(charge=1, spin=1)
    yield mf
    mf.mol.stdout.close()

@pytest.fixture(scope="module")
def h2o_anion_uks():
    # spin-polarized (N+2)-electron reference for hhRPA, 6 alpha and 5 beta electrons
    mf = h2o_b3lyp(charge=-1, spin=1)
    yield mf
    mf.mol.stdout.close()

# ppRPA
def test_pprpa_aa(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks, nvir_act=10)
    pp.kernel(subspace=['aa'])
    pp.analyze()
    assert pp.exci[0][10:13] == pytest.approx([1.21904747, 1.67543516, 1.67748930], abs=1e-5)

def test_pprpa_ab(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks, nvir_act=10)
    pp.kernel(subspace=['ab'])
    pp.analyze()
    assert pp.exci[2][20:23] == pytest.approx([0.91529771, 0.99456249, 1.20050243], abs=1e-5)

def test_pprpa_bb(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks, nvir_act=10)
    pp.kernel(subspace=['bb'])
    pp.analyze()
    assert pp.exci[1][6:9] == pytest.approx([0.91570339, 0.99641030, 1.23454832], abs=1e-5)

def test_pprpa_all(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks, nvir_act=10)
    pp.kernel()
    pp.analyze()
    assert pp.exci[0][10:13] == pytest.approx([1.21904747, 1.67543516, 1.67748930], abs=1e-5)
    assert pp.exci[2][20:23] == pytest.approx([0.91529771, 0.99456249, 1.20050243], abs=1e-5)
    assert pp.exci[1][6:9] == pytest.approx([0.91570339, 0.99641030, 1.23454832], abs=1e-5)

# hhRPA
def test_hhrpa_aa(h2o_anion_uks):
    pp = UppRPADirect(h2o_anion_uks, nvir_act=10, nelec='n+2')
    pp.kernel(subspace=['aa'])
    pp.analyze()
    assert pp.exci[0][12:15] == pytest.approx([-0.85097760, -0.70148499, -0.63465026], abs=1e-5)

def test_hhrpa_ab(h2o_anion_uks):
    pp = UppRPADirect(h2o_anion_uks, nvir_act=10, nelec='n+2')
    pp.kernel(subspace=['ab'])
    pp.analyze()
    assert pp.exci[2][27:30] == pytest.approx([-0.86634263, -0.70646009, -0.63506657], abs=1e-5)

def test_hhrpa_bb(h2o_anion_uks):
    pp = UppRPADirect(h2o_anion_uks, nvir_act=10, nelec='n+2')
    pp.kernel(subspace=['bb'])
    pp.analyze()
    assert pp.exci[1][7:10] == pytest.approx([-1.36801041, -1.31686146, -1.20814546], abs=1e-5)

def test_hhrpa_all(h2o_anion_uks):
    pp = UppRPADirect(h2o_anion_uks, nvir_act=10, nelec='n+2')
    pp.kernel()
    pp.analyze()
    assert pp.exci[0][12:15] == pytest.approx([-0.85097760, -0.70148499, -0.63465026], abs=1e-5)
    assert pp.exci[2][27:30] == pytest.approx([-0.86634263, -0.70646009, -0.63506657], abs=1e-5)
    assert pp.exci[1][7:10] == pytest.approx([-1.36801041, -1.31686146, -1.20814546], abs=1e-5)

# correlation energy
def test_correlation_aa(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks)
    pp.kernel(subspace=['aa'])
    assert pp.ec[0] == pytest.approx(-0.023539954, abs=1e-7)

def test_correlation_ab(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks)
    pp.kernel(subspace=['ab'])
    assert pp.ec[2] == pytest.approx(-0.1061896245, abs=1e-7)

def test_correlation_bb(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks)
    pp.kernel(subspace=['bb'])
    assert pp.ec[1] == pytest.approx(-0.0111385019, abs=1e-7)

def test_correlation_all(h2o_cation_uks):
    pp = UppRPADirect(h2o_cation_uks)
    etot, _, ec = pp.energy_tot()
    assert ec == pytest.approx(-0.1408680803, abs=1e-7)
    assert etot == pytest.approx(-75.6995433844, abs=1e-7)
