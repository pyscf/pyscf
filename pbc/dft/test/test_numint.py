import unittest
import numpy
import numpy as np

from pyscf import gto
from pyscf.dft import rks
import pyscf.dft

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
import pyscf.pbc
pyscf.pbc.DEBUG = False


def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shl_slice=None,
            non0tab=None, out=None, verbose=None):
    gamma_point = kpt is None or abs(kpt).sum() < 1e-9
    aoR = 0
    for L in cell.get_lattice_Ls(cell.nimgs):
        if gamma_point:
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, deriv, relativity,
                                            shl_slice, non0tab, out, verbose)
        else:
            factor = numpy.exp(1j*numpy.dot(kpt,L))
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, deriv, relativity,
                                            shl_slice, non0tab, out, verbose) * factor
    return numpy.asarray(aoR)

def finger(a):
    w = np.cos(np.arange(a.size))
    return np.dot(w, a.ravel())

def make_grids(n):
    L = 60
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [n,n,n]
    cell.nimgs = [0,0,0]

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = {'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.build(False, False)
    grids = gen_grid.UniformGrids(cell)
    grids.build()
    return cell, grids


class KnowValues(unittest.TestCase):
    def test_eval_ao(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.h = np.eye(3) * 2.5
        cell.gs = [10]*3
        cell.atom = [['C', (1., .8, 1.9)],
                     ['C', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        ni = numint._NumInt()
        ao0 = eval_ao(cell, grids.coords)
        ao1 = ni.eval_ao(cell, grids.coords)
        self.assertTrue(numpy.allclose(ao0, ao1, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(finger(ao1), -0.54069672246407219, 8)

        ao0 = eval_ao(cell, grids.coords, deriv=1)
        ao1 = ni.eval_ao(cell, grids.coords, deriv=1)
        self.assertTrue(numpy.allclose(ao0, ao1, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(finger(ao1), 8.8004405892746433, 8)


    def test_eval_mat(self):
        cell, grids = make_grids(30)
        ng = grids.weights.size
        np.random.seed(1)
        rho = np.random.random(ng)
        rho *= 1/np.linalg.norm(rho)
        vrho = np.random.random(ng)
        ao1 = numint.eval_ao(cell, grids.coords)
        mat1 = numint.eval_mat(cell, ao1, grids.weights, rho, vrho)
        w = np.arange(mat1.size) * .01
        self.assertAlmostEqual(np.dot(w,mat1.ravel()), (.14777107967912118+0j), 8)

    def test_eval_ao_kpts(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.h = np.eye(3) * 2.5
        cell.gs = [10]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        np.random.seed(1)
        kpts = np.random.random((4,3))
        ni = numint._KNumInt(kpts)
        ao1 = ni.eval_ao(cell, grids.coords, kpts)
        self.assertAlmostEqual(finger(ao1[0]), (-2.4066959390326477-0.98044994099240701j), 8)
        self.assertAlmostEqual(finger(ao1[1]), (-0.30643153325360639+0.1571658820483913j), 8)
        self.assertAlmostEqual(finger(ao1[2]), (-1.1937974302337684-0.39039259235266233j), 8)
        self.assertAlmostEqual(finger(ao1[3]), (0.17701966968272009-0.20232879692603079j), 8)

    def test_eval_ao_kpt(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.h = np.eye(3) * 2.5
        cell.gs = [10]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        np.random.seed(1)
        kpt = np.random.random(3)
        ni = numint._NumInt()
        ao0 = eval_ao(cell, grids.coords, kpt)
        ao1 = ni.eval_ao(cell, grids.coords, kpt)
        self.assertTrue(numpy.allclose(ao0, ao1, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(finger(ao1), (-2.4066959390326477-0.98044994099240701j), 8)

    def test_nr_rks(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.h = np.eye(3) * 2.5
        cell.gs = [10]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()
        nao = cell.nao_nr()

        np.random.seed(1)
        kpts = np.random.random((2,3))
        dms = np.random.random((2,nao,nao))
        ni = numint._NumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms[0], 0, kpts[0])
        self.assertAlmostEqual(ne, 5.0499199224525153, 8)
        self.assertAlmostEqual(exc, -3.8351805868876574, 8)
        self.assertAlmostEqual(finger(vmat), (-6.7127152097495877+0.18334126894647734j), 8)

        ni = numint._KNumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms, 0, kpts)
        self.assertAlmostEqual(ne, 6.0923292346269742, 8)
        self.assertAlmostEqual(exc, -3.9775343008367003, 8)
        self.assertAlmostEqual(finger(vmat[0]), (-2502.6531823220012-64.707798861780063j), 8)
        self.assertAlmostEqual(finger(vmat[1]), (-2506.9881499848775-125.33971814270384j), 8)

        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', [dms,dms], 0, kpts)
        self.assertAlmostEqual(ne[1], 6.0923292346269742, 8)
        self.assertAlmostEqual(exc[1], -3.9775343008367003, 8)
        self.assertAlmostEqual(finger(vmat[1][0]), (-2502.6531823220012-64.707798861780063j), 8)
        self.assertAlmostEqual(finger(vmat[1][1]), (-2506.9881499848775-125.33971814270384j), 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()

