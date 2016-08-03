#!/usr/bin/env python

import unittest
import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import ft_ao
from pyscf.pbc import tools

cell = pgto.Cell()
cell.atom = '''
He1   1.3    .2       .3
He2    .1    .1      1.1 '''
cell.basis = {'He1': 'sto3g', 'He2': 'ccpvdz'}
cell.gs = (15,)*3
cell.h = numpy.diag([2.2, 1.9, 2.])
cell.build()

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_ft_ao(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords)
        ngs, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i], cell.gs) for i in range(nao)])
        ref = ref.T * (cell.vol/ngs)
        dat = ft_ao.ft_ao(cell, cell.Gv)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 8.4358614794095722e-11, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0.0041669297531642616 , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 5.8677286005879366e-14, 9)

    def test_ft_ao_with_kpts(self):
        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords, kpt=kpt)
        ngs, nao = aoR.shape
        expmikr = numpy.exp(-1j*numpy.dot(coords,kpt))
        ref = numpy.asarray([tools.fftk(aoR[:,i], cell.gs, expmikr) for i in range(nao)])
        ref = ref.T * (cell.vol/ngs)
        dat = ft_ao.ft_ao(cell, cell.Gv, kpt=kpt)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 1.3359899490499813e-10, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0.0042404556036939756 , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 4.8856357999633564e-14, 9)

    def test_ft_aoao(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords)
        ngs, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i].conj()*aoR[:,j], cell.gs)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngs)
        dat = ft_ao.ft_aopair(cell, cell.Gv, hermi=True)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 1.869103994619606e-06 , 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0.02315483195832373   , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 5.4648896424693173e-14, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 4.0352047774658308e-11, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 4.0352047774658308e-11, 9)

    def test_ft_aoao_with_kpts(self):
        numpy.random.seed(1)
        kpti, kptj = numpy.random.random((2,3))
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoi = pdft.numint.eval_ao(cell, coords, kpt=kpti)
        aoj = pdft.numint.eval_ao(cell, coords, kpt=kptj)
        ngs, nao = aoj.shape
        q = kptj - kpti
        ref = numpy.asarray([tools.fftk(aoi[:,i].conj()*aoj[:,j], cell.gs, coords, q)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngs)
        dat = ft_ao.ft_aopair(cell, cell.Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 1.8912693795904546e-06, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0.023225471785938184  , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 3.9231124086361633e-14, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 3.6949758392853562e-11, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 4.1245047267152665e-11, 9)

    def test_ft_aoao_with_kpts(self):
        numpy.random.seed(1)
        kpti, kptj = kpts = numpy.random.random((2,3))
        Gv = cell.get_Gv([5]*3)
        kpt = numpy.random.random(3)
        dat = ft_ao._ft_aopair_kpts(cell, Gv, kpt=kpt, kptjs=kpts)
        self.assertAlmostEqual(finger(dat[0]), (2.3753953914129382-2.5365192689115088j), 9)
        self.assertAlmostEqual(finger(dat[1]), (2.4951510097641840-3.1990956672116355j), 9)
        dat = ft_ao.ft_aopair(cell, Gv)
        self.assertAlmostEqual(finger(dat), (1.2534723618134684+1.830086071817564j), 9)


if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
