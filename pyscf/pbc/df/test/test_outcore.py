import unittest
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
import pyscf.pbc
#pyscf.pbc.DEBUG = False

cell = pgto.Cell()
cell.unit = 'B'
cell.a = numpy.eye(3) * 4.
cell.gs = [5,5,5]
cell.atom = 'He 0 1 1; He 1 1 0'
cell.basis = { 'He': [[0, (0.8, 1.0)],
                      [0, (1.2, 1.0)]] }
cell.verbose = 0
cell.build(0, 0)

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_aux_e2(self):
        tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        numpy.random.seed(1)
        kptij_lst = numpy.random.random((3,2,3))
        kptij_lst[0] = 0
        outcore.aux_e2(cell, cell, tmpfile.name, aosym='s2', comp=1,
                       kptij_lst=kptij_lst, verbose=0)
        refk = incore.aux_e2(cell, cell, aosym='s2', kptij_lst=kptij_lst)
        with h5py.File(tmpfile.name, 'r') as f:
            nao = cell.nao_nr()
            idx = numpy.tril_indices(nao)
            idx = idx[0] * nao + idx[1]
            self.assertTrue(numpy.allclose(refk[0,idx], f['eri_mo/0'].value.T))
            self.assertTrue(numpy.allclose(refk[1], f['eri_mo/1'].value.T))
            self.assertTrue(numpy.allclose(refk[2], f['eri_mo/2'].value.T))

if __name__ == '__main__':
    print("Full Tests for pbc.df.outcore")
    unittest.main()


