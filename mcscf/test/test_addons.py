#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 0,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
)
mfr = scf.RHF(mol)
mfr.scf()
mcr = mcscf.CASSCF(mol, mfr, 4, 4)
mcr.conv_tol_grad = 1e-6
mcr.mc1step()[0]

mfu = scf.UHF(mol)
mfu.scf()
mcu = mcscf.CASSCF(mol, mfu, 4, 4)
mcu.conv_tol_grad = 1e-6
mcu.mc1step()[0]


class KnowValues(unittest.TestCase):
    def test_spin_square(self):
        ss = mcscf.addons.spin_square(mcr)[0]
        self.assertAlmostEqual(ss, 0, 7)
        ss = mcscf.addons.spin_square(mcu)[0]
        self.assertAlmostEqual(ss, 0, 7)

    def test_rcas_natorb(self):
        ci1, mo1, mocc1 = mcscf.addons.cas_natorb(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(mo1)  , 11.4470460817871, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mocc1), 2.59144964144265, 7)

    def test_ucas_natorb(self):
        ci2, mo2, mocc2 = mcscf.addons.cas_natorb(mcu)
        self.assertAlmostEqual(numpy.linalg.norm(mo2)  , 11.4470460817871*numpy.sqrt(2), 7)
        self.assertAlmostEqual(numpy.linalg.norm(mocc2), 2.59144964144265/numpy.sqrt(2), 7)

    def test_map2hf(self):
        idx = mcscf.addons.map2hf(mcr)
        idx0 = zip(*((range(mfr.mo_energy.size),)*2))
        self.assertTrue(numpy.allclose(idx, idx0))

    def test_get_fock(self):
        f1 = mcscf.addons.get_fock(mcr)
        self.assertTrue(numpy.allclose(f1, f1.T))
        self.assertAlmostEqual(numpy.linalg.norm(f1), 23.59747669, 7)
        f1 = mcscf.addons.get_fock(mcu)
        self.assertTrue(numpy.allclose(f1[0], f1[0].T))
        self.assertTrue(numpy.allclose(f1[1], f1[1].T))
        self.assertAlmostEqual(numpy.linalg.norm(f1), 23.59747669*numpy.sqrt(2), 6)

    def test_make_rdm12(self):
        dmr = mcscf.addons.make_rdm1(mcr)
        dm1, dm2 = mcscf.addons.make_rdm12(mcr)
        self.assertTrue(numpy.allclose(dmr, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 5.89864105161976, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 36.3246393695163, 5)

    def test_make_rdm1s(self):
        dm1 = mcscf.addons.make_rdm1s(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 4.17096923, 6)
        dm1 = mcscf.addons.make_rdm1s(mcu)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 4.17096923, 6)

    def test_sort_mo(self):
        mo1 = numpy.arange(mfr.mo_energy.size).reshape(1,-1)
        ref = [[0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        mo2 = mcscf.addons.sort_mo(mcr, mo1, [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, ref))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, (ref,ref)))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [[5,6,7,9],[5,6,8,9]])
        ref1 = [[0, 1, 2, 3, 6, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        self.assertTrue(numpy.allclose(mo2, (ref,ref1)))

    def test_project_init_guess(self):
        print('todo')


if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()

