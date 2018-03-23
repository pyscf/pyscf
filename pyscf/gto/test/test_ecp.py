#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Analytical integration
# J. Chem. Phys. 65, 3826
# J. Chem. Phys. 111, 8778
# J. Comput. Phys. 44, 289
#
# Numerical integration
# J. Comput. Chem. 27, 1009
# Chem. Phys. Lett. 296, 445
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf


class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mol = gto.M(atom='Na 0. 0. 0.;  H  0.  0.  1.',
                    basis={'Na':'lanl2dz', 'H':'sto3g'},
                    ecp = {'Na':'lanl2dz'},
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.45002315563472206, 10)

    def test_bfd(self):
        mol = gto.M(atom='H 0. 0. 0.',
                    basis={'H':'bfd-vdz'},
                    ecp = {'H':'bfd-pp'},
                    spin = 1,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.499045, 6)

        mol = gto.M(atom='Na 0. 0. 0.',
                    basis={'Na':'bfd-vtz'},
                    ecp = {'Na':'bfd-pp'},
                    spin = 1,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.181799, 6)

        mol = gto.M(atom='Mg 0. 0. 0.',
                    basis={'Mg':'bfd-vtz'},
                    ecp = {'Mg':'bfd-pp'},
                    spin = 0,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.784579, 6)

#        mol = gto.M(atom='Ne 0. 0. 0.',
#                    basis={'Ne':'bfd-vdz'},
#                    ecp = {'Ne':'bfd-pp'},
#                    verbose=0)
#        mf = scf.RHF(mol)
#        self.assertAlmostEqual(mf.kernel(), -34.709059, 6)


if __name__ == '__main__':
    print("Full Tests for ECP")
    unittest.main()

