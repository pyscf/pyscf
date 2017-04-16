import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import incore
import pyscf.pbc
pyscf.pbc.DEBUG = False


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_aux_e2(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        cell.a = numpy.eye(3) * 3.
        cell.gs = numpy.array([20,20,20])
        cell.atom = 'He 0 1 1; He 1 1 0'
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [0, (1.2, 1.0)]] }
        cell.verbose = 0
        cell.build(0, 0)
        auxcell = incore.format_aux_basis(cell)
        cell.nimgs = auxcell.nimgs = [3,3,3]
        a1 = incore.aux_e2(cell, auxcell, 'cint3c1e_sph')
        self.assertAlmostEqual(finger(a1), 0.1208944790152819, 9)
        a2 = incore.aux_e2(cell, auxcell, 'cint3c1e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, axis=0).reshape(a1.shape)))

        numpy.random.seed(3)
        kpti_kptj = [numpy.random.random(3)]*2
        a1 = incore.aux_e2(cell, auxcell, 'cint3c1e_sph', kpti_kptj=kpti_kptj)
        self.assertAlmostEqual(finger(a1), -0.073719031689332651-0.054002639392614758j, 9)
        a2 = incore.aux_e2(cell, auxcell, 'cint3c1e_sph', aosym='s2ij', kpti_kptj=kpti_kptj)
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, 1, axis=0).reshape(a1.shape)))

        numpy.random.seed(1)
        kpti_kptj = numpy.random.random((2,3))
        a1 = incore.aux_e2(cell, auxcell, 'cint3c1e_sph', kpti_kptj=kpti_kptj)
        self.assertAlmostEqual(finger(a1), 0.039329191948685879-0.039836453846241987j, 9)

if __name__ == '__main__':
    print("Full Tests for pbc.df.incore")
    unittest.main()

