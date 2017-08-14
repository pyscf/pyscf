import unittest
import numpy
import tempfile
from pyscf import lib, gto

class KnowValues(unittest.TestCase):
    def test_save_load_mol(self):
        mol = gto.M(atom=[['H', (0,0,i)] for i in range(8)],
                    basis='sto3g')
        fchk = tempfile.NamedTemporaryFile()
        lib.chkfile.save_mol(mol, fchk.name)
        mol1 = lib.chkfile.load_mol(fchk.name)
        self.assertTrue(numpy.all(mol1._atm == mol._atm))
        self.assertTrue(numpy.all(mol1._bas == mol._bas))
        self.assertTrue(numpy.all(mol1._env == mol._env))

    def test_save_load_arrays(self):
        fchk = tempfile.NamedTemporaryFile()
        a = numpy.eye(3)
        lib.chkfile.save(fchk.name, 'a', a)
        self.assertTrue(numpy.all(a == lib.chkfile.load(fchk.name, 'a')))

        a = [numpy.eye(3), numpy.eye(4)]
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue(isinstance(dat, list))
        self.assertTrue(numpy.all(a[1] == dat[1]))

        a = [[numpy.eye(4), numpy.eye(4)]]*2
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue(isinstance(dat, list))
        self.assertTrue(isinstance(dat[0], list))

        a = {'x':[numpy.eye(4), numpy.eye(4)],
             'y':[numpy.eye(4)]}
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue('x' in dat)
        self.assertTrue('y' in dat)

if __name__ == "__main__":
    print("Full Tests for lib.chkfile")
    unittest.main()
