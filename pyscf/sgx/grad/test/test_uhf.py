from pyscf.sgx.sgx import sgx_fit
from pyscf import gto, scf, lib
import numpy
import unittest


ALL_SETTINGS = [
    [False, False, False, False, False],
    [True, False, False, False, False],
    [True, True, False, False, False],
    [True, False, True, False, True],
    [True, True, True, True, False],
    [True, True, True, True, True],
]
ALL_PRECISIONS = [5, 6, 3, 6, 5, 6]


def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)],
    ])
    mol.basis = '6-31g'
    mol.spin = 1
    mol.charge = 1
    mol.build()


def _set_df_args(mf, dfj, fit_ovlp, optk, dm_screen, symm_fit):
    if dfj:
        mf.with_df.grids_level_f = 1
    else:
        mf.with_df.grids_level_f = 2
    mf.with_df.dfj = dfj
    mf.with_df.fit_ovlp = fit_ovlp
    mf.with_df.optk = optk
    if dm_screen:
        mf.with_df.sgx_tol_energy = 1e-12
    else:
        mf.with_df.sgx_tol_energy = None
        mf.with_df.sgx_tol_potential = None
    mf.with_df._symm_ovlp_fit = symm_fit
    mf.conv_tol = 1e-12


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):

    def _check_finite_diff_grad(self, df_settings, order, sgr=True):
        mf = sgx_fit(scf.UHF(mol))
        _set_df_args(mf, *df_settings)
        g = mf.nuc_grad_method().set(sgx_grid_response=sgr, grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        delta = 1e-4
        e1 = mf_scanner(mol1.set_geom_(
            f'O  0. 0. {delta:f}; 1  0. -0.757 0.587; 1  0. 0.757 0.587'
        ))
        e2 = mf_scanner(mol1.set_geom_(
            f'O  0. 0. -{delta:f}; 1  0. -0.757 0.587; 1  0. 0.757 0.587'
        ))
        if sgr:
            # Forces will not sum to zero unless sgx_grid_response=True
            self.assertAlmostEqual(numpy.abs(g.sum(axis=0)).sum(), 0, 13)
        self.assertAlmostEqual(g[0,2], (e1-e2)/(2*delta)*lib.param.BOHR, order)

    def test_finite_diff_grad(self):
        self._check_finite_diff_grad(ALL_SETTINGS[0], ALL_PRECISIONS[0])
        self._check_finite_diff_grad(ALL_SETTINGS[1], ALL_PRECISIONS[1])
        self._check_finite_diff_grad(ALL_SETTINGS[2], ALL_PRECISIONS[2])
        self._check_finite_diff_grad(ALL_SETTINGS[3], ALL_PRECISIONS[3])
        self._check_finite_diff_grad(ALL_SETTINGS[4], ALL_PRECISIONS[4])
        self._check_finite_diff_grad(ALL_SETTINGS[5], ALL_PRECISIONS[5])

    def test_finite_diff_noresponse(self):
        self._check_finite_diff_grad(ALL_SETTINGS[0], 3, False)
        self._check_finite_diff_grad(ALL_SETTINGS[5], 3, False)


if __name__ == '__main__':
    unittest.main()
