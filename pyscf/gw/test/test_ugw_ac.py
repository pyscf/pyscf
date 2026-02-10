import pytest

from pyscf import gto, dft, scf
from pyscf.gw.ugw_ac import UGWAC

@pytest.fixture
def h2o_cation_uhf():
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

def test_ugw_ac(h2o_cation_uhf):
    gw = UGWAC(h2o_cation_uhf)
    gw.orbs = range(2, 8)
    gw.kernel()
    assert gw.mo_energy[0][4] == pytest.approx(-1.02679347, abs=1e-5)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15525786, abs=1e-5)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99401046, abs=1e-5)
    assert gw.mo_energy[1][4] == pytest.approx(-0.42543725, abs=1e-5)
