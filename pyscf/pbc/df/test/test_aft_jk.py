# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc.df import aft, aft_jk


cell = gto.Cell()
cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
cell.a = numpy.eye(3) * 2.5
cell.mesh = [21] * 3
cell.build()


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_jk(self):
        mf0 = scf.RHF(cell)
        dm = mf0.get_init_guess()

        mydf = aft.AFTDF(cell)
        mydf.mesh = [11]*3
        vj, vk = mydf.get_jk(dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 3.0455881073561235, 9)
        self.assertAlmostEqual(ek1, 7.7905480251964629, 9)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mydf.get_jk(dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 12.234106555081793, 9)
        self.assertAlmostEqual(ek1, 43.988705494650802, 9)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 12.233546641482697, 9)
        self.assertAlmostEqual(ek1, 43.946958026023722, 9)

    def test_aft_j(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = aft.AFTDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        vj = aft_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.93946193432413905+0.00010862804196223034j)/4, 9)
        self.assertAlmostEqual(finger(vj[1]), (0.94866073525335626+0.005571199307452865j)  /4, 9)
        self.assertAlmostEqual(finger(vj[2]), (1.1492194255929766+0.0093705761598793739j)  /4, 9)
        self.assertAlmostEqual(finger(vj[3]), (1.1397493412770023+0.010731970529096637j)   /4, 9)

    def test_aft_k(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = aft.AFTDF(cell)
        mydf.kpts = kpts
        vk = aft_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (4.3373802352168278-0.062977052131451577j)/8, 9)
        self.assertAlmostEqual(finger(vk[1]), (2.878809181709983+0.028843869853690692j) /8, 9)
        self.assertAlmostEqual(finger(vk[2]), (3.7027622609953061-0.052034330663180237j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (5.0939994842559422+0.060094478876149444j)/8, 9)
        self.assertAlmostEqual(finger(vk[4]), (4.2942087551592651-0.061138484763336887j)/8, 9)
        self.assertAlmostEqual(finger(vk[5]), (3.9689429688683679+0.048471952758750547j)/8, 9)
        self.assertAlmostEqual(finger(vk[6]), (3.6342630872923456-0.054892635365850449j)/8, 9)
        self.assertAlmostEqual(finger(vk[7]), (3.3483735224533548+0.040877095049528467j)/8, 9)

    def test_aft_k1_high_cost(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = aft.AFTDF(cell)
        mydf.kpts = kpts
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = aft_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (8.7518173818250702-0.11793770445839372j) /8, 9)
        self.assertAlmostEqual(finger(vk[1]), (5.7682393685317894+0.069482280306391239j)/8, 9)
        self.assertAlmostEqual(finger(vk[2]), (7.1890462727492324-0.088727079644645671j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (10.08358152800003+0.1278144339422369j   )/8, 9)
        self.assertAlmostEqual(finger(vk[4]), (8.393281242945573-0.099410704957774876j) /8, 9)
        self.assertAlmostEqual(finger(vk[5]), (7.9413682328898769+0.1015563120870652j)  /8, 9)
        self.assertAlmostEqual(finger(vk[6]), (7.3743790120272408-0.096290683129384574j)/8, 9)
        self.assertAlmostEqual(finger(vk[7]), (6.8144379626901443+0.08071261392857812j) /8, 9)



if __name__ == '__main__':
    print("Full Tests for aft jk")
    unittest.main()

