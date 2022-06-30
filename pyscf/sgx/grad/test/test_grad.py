from pyscf.sgx.sgx import sgx_fit
from pyscf import gto, scf, lib
import unittest

def setUpModule():
    global mol, mf1, mf2, mf3
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)],
    ])
    mol.basis = '6-31g'
    mol.build()

    mf1 = sgx_fit(scf.RHF(mol))
    mf1.with_df.grids_level_f = 2
    mf1.with_df.dfj = True
    mf1.with_df.fit_ovlp = False
    mf1.conv_tol = 1e-14
    mf1.kernel()

    mf2 = sgx_fit(scf.RHF(mol))
    mf2.with_df.grids_level_f = 2
    mf2.with_df.dfj = False
    mf2.with_df.fit_ovlp = False
    mf2.conv_tol = 1e-14
    mf2.kernel()

    mf3 = sgx_fit(scf.UKS(mol).set(xc='PBE0'))
    mf3.with_df.grids_level_f = 2
    mf3.with_df.dfj = True
    mf3.with_df.fit_ovlp = False
    mf3.conv_tol = 1e-14
    mf3.kernel()

def tearDownModule():
    global mol, mf1, mf2, mf3
    mol.stdout.close()
    del mol, mf1, mf2, mf3


class KnownValues(unittest.TestCase):

    def test_finite_diff_grad(self):
        for mf, order in zip([mf1, mf2, mf3], [6, 5, 6]):
            g = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True).kernel()
            mol1 = mol.copy()
            mf_scanner = mf.as_scanner()
            e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
            e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
            self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, order)

if __name__ == '__main__':
    unittest.main()
