from pyscf.sgx.sgx import sgx_fit
from pyscf import gto, scf, lib
from pyscf.lib import logger
import numpy
import unittest


ATOM = [
    ["H" , (10. , 0. , 1.804)],
    ["F" , (10. , 0. , 0.   )],
    ["O" , (0. , 0.     , 0.)],
    ["H" , (0. , -0.757 , 0.587)],
    ["H" , (0. , 0.757  , 0.587)],
]


def setUpModule():
    global mol, mol2
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend(ATOM)
    mol.basis = '6-31g'
    mol.max_memory = 2000
    mol.build()
    mol2 = mol.copy()
    mol2.spin = 1
    mol2.charge = 1
    mol2.build()


def _set_df_args(mf, fit_ovlp, optk):
    mf.with_df.grids_level_f = 1
    mf.with_df.dfj = True
    mf.with_df.fit_ovlp = fit_ovlp
    mf.with_df.optk = optk
    mf.with_df.use_dm_screening = True
    mf.with_df.sgx_tol_potential = 1e-6
    mf.with_df.sgx_tol_energy = 1e-12
    mf.rebuild_nsteps = 1
    mf.conv_tol = 1e-12


def tearDownModule():
    global mol, mol2
    mol.stdout.close()
    mol2.stdout.close()
    del mol, mol2


class KnownValues(unittest.TestCase):

    def _check_finite_diff_grad(self, spinpol, fit_ovlp, optk):
        t0 = logger.perf_counter()
        if spinpol:
            mf = sgx_fit(scf.UHF(mol2))
        else:
            mf = sgx_fit(scf.RHF(mol))
        _set_df_args(mf, fit_ovlp, optk)
        mf.kernel()
        t1 = logger.perf_counter()
        g = mf.nuc_grad_method().set(sgx_grid_response=True, grid_response=True).kernel()
        t2 = logger.perf_counter()
        if spinpol:
            mol1 = mol2.copy()
        else:
            mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        delta = 1e-5
        atomp = [a for a in ATOM]
        atomp[2] = ["O" , (0., 0., delta)]
        atomm = [a for a in ATOM]
        atomm[2] = ["O" , (0., 0., -delta)]
        e1 = mf_scanner(mol1.set_geom_(atomp))
        e2 = mf_scanner(mol1.set_geom_(atomm))
        t3 = logger.perf_counter()
        self.assertAlmostEqual(g[2,2], (e1-e2)/(2*delta)*lib.param.BOHR, 6)
        self.assertAlmostEqual(numpy.abs(g.sum(axis=0)).sum(), 0, 13)
        return mf.e_tot, (t3 - t2) + (t1 - t0), t2 - t1, mf.cycles

    def test_rhf_dm_screening(self):
        e0, te0, tg0, c0 = self._check_finite_diff_grad(False, False, False)
        e1, te1, tg1, c1 = self._check_finite_diff_grad(False, False, True)
        e2, te2, tg2, c2 = self._check_finite_diff_grad(False, True, True)
        self.assertAlmostEqual(e1, e0, 8)
        self.assertAlmostEqual(e2, e1, 3)
        msg = "Times for energy and gradients, in s\n"
        msg = msg + "No Optk %f %f %f\n" % (te0, tg0, c0)
        msg = msg + "Optk %f %f %f\n" % (te1, tg1, c1)
        msg = msg + "Optk w/ovlp fit %f %f %f\n" % (te2, tg2, c2)
        print(msg)

    def test_uhf_dm_screening(self):
        e0, te0, tg0, c0 = self._check_finite_diff_grad(True, False, False)
        e1, te1, tg1, c1 = self._check_finite_diff_grad(True, False, True)
        e2, te2, tg2, c2 = self._check_finite_diff_grad(True, True, True)
        self.assertAlmostEqual(e1, e0, 8)
        self.assertAlmostEqual(e2, e1, 3)
        msg = "Times for energy and gradients, in s\n"
        msg = msg + "No Optk %f %f %f\n" % (te0, tg0, c0)
        msg = msg + "Optk %f %f %f\n" % (te1, tg1, c1)
        msg = msg + "Optk w/ovlp fit %f %f %f\n" % (te2, tg2, c2)
        print(msg)


if __name__ == '__main__':
    unittest.main()
