import unittest
import numpy
from pyscf.pbc import gto, scf
cell = gto.M(atom='''
C 4.826006352031   3.412501814582   8.358888185226
C 0.689429478862   0.487500259226   1.194126883604
             ''',
a='''
4.136576868, 0.000000000, 2.388253772
1.378858962, 3.900002074, 2.388253772
0.000000000, 0.000000000, 4.776507525
             ''',
unit='B',
precision=1e-14,
basis='gth-tzv2p',
pseudo='gth-lda',
gs=[7]*3,
verbose=0)

class KnownValues(unittest.TestCase):
    def test_rcut(self):
        rcut_ref = cell.rcut
        #print rcut_ref
        kpts = cell.make_kpts([2,2,2])
        rcut_ref = rcut_ref
        t0 = numpy.asarray(cell.pbc_intor('int1e_kin_sph', hermi=1, kpts=kpts))
        s0 = numpy.asarray(cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts))
        for i in range(1, 10):
            prec = 1e-13 * 10**i
            cell.rcut = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
            t1 = numpy.asarray(cell.pbc_intor('int1e_kin_sph', hermi=1, kpts=kpts))
            s1 = numpy.asarray(cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts))
            #print prec, cell.rcut, abs(t1-t0).max(), abs(s1-s0).max()
            self.assertTrue(abs(t1-t0).max() < prec*1e-1)
            self.assertTrue(abs(s1-s0).max() < prec*1e-2)

if __name__ == '__main__':
    print("Test rcut and the errorsin pbc.gto.cell")
    unittest.main()

