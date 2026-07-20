import unittest
import numpy as np
from pyscf.pbc import gto, dft
from pyscf.pbc.scf.addons import slab_dipole_correction


def build_cell(polar):
    a, c = 4.0, 12.0
    if polar:
        atom = f"F {a/2} {a/2} 5.00; H {a/2} {a/2} 5.92"
    else:
        atom = f"H {a/2} {a/2} 5.60; H {a/2} {a/2} 6.40"
    cell = gto.Cell()
    cell.atom = atom
    cell.a = np.diag([a, a, c])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.unit = "A"
    cell.dimension = 3
    cell.verbose = 0
    cell.build()
    return cell

def run(cell, corrected):
    mf = dft.KRKS(cell, cell.make_kpts([1, 1, 1])).density_fit()
    mf.xc = "pbe"
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    if corrected:
        slab_dipole_correction(mf, dir_idx=2)
    e = mf.kernel()
    return mf, e


class SlabDipoleCorrectionTest(unittest.TestCase):
    def _run_pair(self, polar):
        cell = build_cell(polar)
        mf0, e0 = run(cell, corrected=False)
        mf1, e1 = run(cell, corrected=True)
        self.assertTrue(mf0.converged, "uncorrected SCF did not converge")
        self.assertTrue(mf1.converged, "corrected SCF did not converge")
        return e1 - e0

    def test_polar_slab(self):
        """A polar slab has a sizeable dipole and the correction shifts E."""
        de = self._run_pair(polar=True)
        self.assertGreater(abs(de), 1e-5,
                           "correction had no effect on a polar slab")

    def test_nonpolar_slab(self):
        """A symmetric slab has ~zero dipole and the correction is ~zero."""
        de = self._run_pair(polar=False)
        self.assertLess(abs(de), 1e-4,
                        "correction should be ~zero for a non-polar slab")


if __name__ == "__main__":
    unittest.main()
