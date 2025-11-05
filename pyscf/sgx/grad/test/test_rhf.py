from pyscf.sgx.sgx import sgx_fit
from pyscf import gto, scf, lib
import unittest


USE_OPTK = True


def setUpModule():
    global mol, mf1, mf2, mf3, mf4, mf5
    mol = gto.Mole()
    mol.verbose = 3
    #mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)],
    ])
    mol.basis = '6-31g'
    mol.max_memory = 2000
    mol.build()

    mf1 = sgx_fit(scf.RHF(mol))
    mf1.with_df.grids_level_f = 1
    mf1.with_df.dfj = True
    mf1.with_df.fit_ovlp = True
    mf1.with_df._symm_ovlp_fit = True
    mf1.with_df.use_dm_screening = True
    mf1.with_df.optk = USE_OPTK
    mf1.conv_tol = 1e-14
    mf1.kernel()

    mf2 = sgx_fit(scf.RHF(mol))
    mf2.with_df.grids_level_f = 2
    mf2.with_df.dfj = False
    mf2.with_df.fit_ovlp = False
    mf2.with_df.optk = False
    mf2.conv_tol = 1e-14
    mf2.kernel()

    mf3 = sgx_fit(scf.RKS(mol).set(xc='PBE0'))
    mf3.with_df.grids_level_f = 1
    mf3.with_df.dfj = True
    mf3.with_df.fit_ovlp = True
    mf3.with_df._symm_ovlp_fit = False
    mf3.with_df.use_dm_screening = True
    mf3.with_df.optk = USE_OPTK
    mf3.conv_tol = 1e-14
    mf3.kernel()

    mf4 = sgx_fit(scf.RKS(mol).set(xc='WB97X'))
    mf4.with_df.grids_level_f = 1
    mf4.with_df.dfj = True
    mf4.with_df.fit_ovlp = True
    mf4.with_df._symm_ovlp_fit = True
    mf4.with_df.use_dm_screening = True
    mf4.with_df.optk = USE_OPTK
    mf4.conv_tol = 1e-14
    mf4.kernel()

    mf5 = sgx_fit(scf.RKS(mol).set(xc='HSE06'))
    mf5.with_df.grids_level_f = 1
    mf5.with_df.dfj = True
    mf5.with_df.fit_ovlp = True
    mf5.with_df._symm_ovlp_fit = True
    mf5.with_df.use_dm_screening = True
    mf5.with_df.optk = USE_OPTK
    mf5.conv_tol = 1e-14
    mf5.kernel()

def tearDownModule():
    global mol, mf1, mf2, mf3, mf4, mf5
    mol.stdout.close()
    del mol, mf1, mf2, mf3, mf4, mf5


class KnownValues(unittest.TestCase):

    def _check_finite_diff_grad(self, mf, order):
        g = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        delta = 1e-5
        e1 = mf_scanner(mol1.set_geom_(f'O  0. 0. {delta:f}; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_(f'O  0. 0. -{delta:f}; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/(2*delta)*lib.param.BOHR, order)

    def test_finite_diff_grad(self):
        self._check_finite_diff_grad(mf1, 6)
        self._check_finite_diff_grad(mf2, 5)
        self._check_finite_diff_grad(mf3, 6)
        self._check_finite_diff_grad(mf4, 6)
        self._check_finite_diff_grad(mf5, 6)

if __name__ == '__main__':
    unittest.main()
