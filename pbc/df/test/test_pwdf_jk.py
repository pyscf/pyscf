import unittest
import numpy
import numpy as np

from pyscf.pbc import gto as pgto
import pyscf.pbc.dft as pdft
from pyscf.pbc.df import pwdf, pwdf_jk


cell = pgto.Cell()
cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
cell.a = np.eye(3) * 2.5
cell.gs = [10] * 3
cell.build()


def finger(a):
    w = np.cos(np.arange(a.size))
    return np.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_pwdf_j(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = pwdf.PWDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.auxbasis = 'weigend'
        vj = pwdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.93946193432413905+0.00010862804196223034j)/4, 9)
        self.assertAlmostEqual(finger(vj[1]), (0.94866073525335626+0.005571199307452865j)  /4, 9)
        self.assertAlmostEqual(finger(vj[2]), (1.1492194255929766+0.0093705761598793739j)  /4, 9)
        self.assertAlmostEqual(finger(vj[3]), (1.1397493412770023+0.010731970529096637j)   /4, 9)

    def test_pwdf_k(self):
        kpts = cell.make_kpts((2,2,2))

        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = pwdf.PWDF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        vk = pwdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (4.3373802352168278-0.062977052131451577j)/8, 9)
        self.assertAlmostEqual(finger(vk[1]), (2.878809181709983+0.028843869853690692j) /8, 9)
        self.assertAlmostEqual(finger(vk[2]), (3.7027622609953061-0.052034330663180237j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (5.0939994842559422+0.060094478876149444j)/8, 9)
        self.assertAlmostEqual(finger(vk[4]), (4.2942087551592651-0.061138484763336887j)/8, 9)
        self.assertAlmostEqual(finger(vk[5]), (3.9689429688683679+0.048471952758750547j)/8, 9)
        self.assertAlmostEqual(finger(vk[6]), (3.6342630872923456-0.054892635365850449j)/8, 9)
        self.assertAlmostEqual(finger(vk[7]), (3.3483735224533548+0.040877095049528467j)/8, 9)

        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = pwdf_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (8.7518173818250702-0.11793770445839372j) /8, 9)
        self.assertAlmostEqual(finger(vk[1]), (5.7682393685317894+0.069482280306391239j)/8, 9)
        self.assertAlmostEqual(finger(vk[2]), (7.1890462727492324-0.088727079644645671j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (10.08358152800003+0.1278144339422369j   )/8, 9)
        self.assertAlmostEqual(finger(vk[4]), (8.393281242945573-0.099410704957774876j) /8, 9)
        self.assertAlmostEqual(finger(vk[5]), (7.9413682328898769+0.1015563120870652j)  /8, 9)
        self.assertAlmostEqual(finger(vk[6]), (7.3743790120272408-0.096290683129384574j)/8, 9)
        self.assertAlmostEqual(finger(vk[7]), (6.8144379626901443+0.08071261392857812j) /8, 9)



if __name__ == '__main__':
    print("Full Tests for pwdf jk")
    unittest.main()

