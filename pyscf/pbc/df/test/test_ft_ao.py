#!/usr/bin/env python
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
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import ft_ao
from pyscf.pbc import tools
from pyscf import lib

def setUpModule():
    global cell, cell1
    cell = pgto.Cell()
    cell.atom = '''
    He1   1.3    .2       .3
    He2    .1    .1      1.1 '''
    cell.basis = {'He1': 'sto3g', 'He2': 'ccpvdz'}
    cell.mesh = (31,)*3
    cell.a = numpy.diag([2.2, 1.9, 2.])
    cell.build()

    cell1 = pgto.Cell()
    cell1.atom = '''
    He   1.3    .2       .3
    He    .1    .1      1.1 '''
    cell1.basis = {'He': [[0, [0.8, 1]],
                          [1, [0.6, 1]]
                         ]}
    cell1.mesh = [17]*3
    cell1.a = numpy.array(([2.0,  .9, 0. ],
                           [0.1, 1.9, 0.4],
                           [0.8, 0  , 2.1]))
    cell1.build()

def tearDownModule():
    global cell, cell1
    del cell, cell1

class KnownValues(unittest.TestCase):
    def test_ft_ao(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords)
        ngrids, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i], cell.mesh) for i in range(nao)])
        ref = ref.T * (cell.vol/ngrids)
        dat = ft_ao.ft_ao(cell, cell.Gv)
        self.assertAlmostEqual(abs(ref[:,0]-dat[:,0])  .max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,1]-dat[:,1])  .max(), 0, 3)
        self.assertAlmostEqual(abs(ref[:,2:]-dat[:,2:]).max(), 0, 8)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoR = pdft.numint.eval_ao(cell1, coords)
        ngrids, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i], cell1.mesh) for i in range(nao)])
        ref = ref.T * (cell1.vol/ngrids)
        dat = ft_ao.ft_ao(cell1, cell1.Gv)
        self.assertAlmostEqual(abs(ref[:,0]-dat[:,0])  .max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,1]-dat[:,1])  .max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,2:]-dat[:,2:]).max(), 0, 8)

    def test_ft_ao_with_kpts(self):
        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords, kpt=kpt)
        ngrids, nao = aoR.shape
        expmikr = numpy.exp(-1j*numpy.dot(coords,kpt))
        ref = numpy.asarray([tools.fftk(aoR[:,i], cell.mesh, expmikr) for i in range(nao)])
        ref = ref.T * (cell.vol/ngrids)
        dat = ft_ao.ft_ao(cell, cell.Gv, kpt=kpt)
        self.assertAlmostEqual(abs(ref[:,0]-dat[:,0])  .max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,1]-dat[:,1])  .max(), 0, 3)
        self.assertAlmostEqual(abs(ref[:,2:]-dat[:,2:]).max(), 0, 8)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoR = pdft.numint.eval_ao(cell1, coords, kpt=kpt)
        ngrids, nao = aoR.shape
        expmikr = numpy.exp(-1j*numpy.dot(coords,kpt))
        ref = numpy.asarray([tools.fftk(aoR[:,i], cell1.mesh, expmikr) for i in range(nao)])
        ref = ref.T * (cell1.vol/ngrids)
        dat = ft_ao.ft_ao(cell1, cell1.Gv, kpt=kpt)
        self.assertAlmostEqual(abs(ref[:,0]-dat[:,0])  .max(), 0, 5)
        self.assertAlmostEqual(abs(ref[:,1]-dat[:,1])  .max(), 0, 3)
        self.assertAlmostEqual(abs(ref[:,2:]-dat[:,2:]).max(), 0, 3)

    def test_ft_aoao(self):
        #coords = pdft.gen_grid.gen_uniform_grids(cell)
        #aoR = pdft.numint.eval_ao(cell, coords)
        #ngrids, nao = aoR.shape
        #ref = numpy.asarray([tools.fft(aoR[:,i].conj()*aoR[:,j], cell.mesh)
        #                     for i in range(nao) for j in range(nao)])
        #ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngrids)
        #dat = ft_ao.ft_aopair(cell, cell.Gv, aosym='s1hermi')
        #self.assertAlmostEqual(abs(ref[:,0,0]-dat[:,0,0])    .max(), 0, 5)
        #self.assertAlmostEqual(abs(ref[:,1,1]-dat[:,1,1])    .max(), 0, 4)
        #self.assertAlmostEqual(abs(ref[:,2:,2:]-dat[:,2:,2:]).max(), 0, 9)
        #self.assertAlmostEqual(abs(ref[:,0,2:]-dat[:,0,2:])  .max(), 0, 9)
        #self.assertAlmostEqual(abs(ref[:,2:,0]-dat[:,2:,0])  .max(), 0, 9)
        #idx = numpy.tril_indices(nao)
        #ref = dat[:,idx[0],idx[1]]
        #dat = ft_ao.ft_aopair(cell, cell.Gv, aosym='s2')
        #self.assertAlmostEqual(abs(dat-ref).sum(), 0, 9)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        b = cell1.reciprocal_vectors()
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(lib.fp(dat), 1.5666516306798806+1.953555017583245j, 7)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s2', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(lib.fp(dat), -0.85276967757297917+1.0378751267506394j, 7)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1hermi', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(lib.fp(dat), 1.5666516306798806+1.953555017583245j, 7)
        aoR = pdft.numint.eval_ao(cell1, coords)
        ngrids, nao = aoR.shape
        aoaoR = numpy.einsum('pi,pj->ijp', aoR, aoR)
        ref = tools.fft(aoaoR.reshape(nao*nao,-1), cell1.mesh)
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref[:,0,0]-dat[:,0,0])    .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,1,1]-dat[:,1,1])    .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,2:,2:]-dat[:,2:,2:]).max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,0,2:]-dat[:,0,2:])  .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,2:,0]-dat[:,2:,0])  .max(), 0, 7)
        idx = numpy.tril_indices(nao)
        ref = dat[:,idx[0],idx[1]]
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s2')
        self.assertAlmostEqual(abs(dat-ref).sum(), 0, 9)

    def test_ft_aoao_pdotp(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', intor='GTO_ft_pdotp_sph')
        self.assertAlmostEqual(lib.fp(dat), 5.7858606710458078-8.654809509773056j, 6)
        aoR = pdft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        aoaoR = numpy.einsum('xpi,xpj->ijp', aoR[1:4], aoR[1:4])
        ref = tools.fft(aoaoR.reshape(nao*nao,-1), cell1.mesh)
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 6)

    def test_ft_aoao_pxp(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', intor='GTO_ft_pxp_sph', comp=3)
        self.assertAlmostEqual(lib.fp(dat), (6.4124798727215779-10.673712733378771j), 6)
        aoR = pdft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        aox, aoy, aoz = aoR[1:]
        aoaoR =(numpy.einsum('pi,pj->ijp', aoy, aoz) - numpy.einsum('pi,pj->ijp', aoz, aoy),
                numpy.einsum('pi,pj->ijp', aoz, aox) - numpy.einsum('pi,pj->ijp', aox, aoz),
                numpy.einsum('pi,pj->ijp', aox, aoy) - numpy.einsum('pi,pj->ijp', aoy, aox))
        ref = tools.fft(numpy.array(aoaoR).reshape(3*nao*nao,-1), cell1.mesh)
        ref = ref.reshape(3,nao,nao,-1).transpose(0,3,1,2) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 7)

    def test_ft_aoao_with_kpts_high_cost(self):
        numpy.random.seed(1)
        kpti, kptj = numpy.random.random((2,3))
        dat = ft_ao.ft_aopair(cell, cell.Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(lib.fp(dat), -0.80184732435570638+2.4078835207597176j, 7)
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoi = pdft.numint.eval_ao(cell, coords, kpt=kpti)
        aoj = pdft.numint.eval_ao(cell, coords, kpt=kptj)
        ngrids, nao = aoj.shape
        q = kptj - kpti
        expmikr = numpy.exp(-1j*numpy.dot(coords,q))
        ref = numpy.asarray([tools.fftk(aoi[:,i].conj()*aoj[:,j], cell.mesh, expmikr)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngrids)
        self.assertAlmostEqual(abs(ref[:,0,0]-dat[:,0,0])    .max(), 0, 5)
        self.assertAlmostEqual(abs(ref[:,1,1]-dat[:,1,1])    .max(), 0, 2)
        self.assertAlmostEqual(abs(ref[:,2:,2:]-dat[:,2:,2:]).max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,0,2:]-dat[:,0,2:])  .max(), 0, 8)
        self.assertAlmostEqual(abs(ref[:,2:,0]-dat[:,2:,0])  .max(), 0, 8)

    def test_ft_aoao_pair_vs_fft(self):
        numpy.random.seed(1)
        kpti, kptj = numpy.random.random((2,3))
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoi = pdft.numint.eval_ao(cell1, coords, kpt=kpti)
        aoj = pdft.numint.eval_ao(cell1, coords, kpt=kptj)
        ngrids, nao = aoj.shape
        q = kptj - kpti
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, kpti_kptj=(kpti,kptj), q=q)
        self.assertAlmostEqual(lib.fp(dat), 0.72664436503332241+3.2542145296611373j, 7)
        expmikr = numpy.exp(-1j*numpy.dot(coords,q))
        ref = numpy.asarray([tools.fftk(aoi[:,i].conj()*aoj[:,j], cell1.mesh, expmikr)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref[:,0,0]-dat[:,0,0])    .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,1,1]-dat[:,1,1])    .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,2:,2:]-dat[:,2:,2:]).max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,0,2:]-dat[:,0,2:])  .max(), 0, 7)
        self.assertAlmostEqual(abs(ref[:,2:,0]-dat[:,2:,0])  .max(), 0, 7)

    def test_ft_aoao_with_kpts1(self):
        numpy.random.seed(1)
        kpti, kptj = kpts = numpy.random.random((2,3))
        Gv = cell.get_Gv([11]*3)
        q = numpy.random.random(3)
        dat = ft_ao.ft_aopair_kpts(cell, Gv, q=q, kptjs=kpts)
        self.assertAlmostEqual(lib.fp(dat[0]), (2.3753953914129382-2.5365192689115088j), 8)
        self.assertAlmostEqual(lib.fp(dat[1]), (2.4951510097641840-3.1990956672116355j), 8)
        dat = ft_ao.ft_aopair(cell, Gv)
        self.assertAlmostEqual(lib.fp(dat), (1.2534723618134684+1.830086071817564j), 7)

    def test_ft_aoao1(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 5
        n = 18
        cell.mesh = numpy.array([n,n,n])
        cell.atom = '''C    1.3    .2       .3
                       C     .1    .1      1.1
                       '''
        cell.basis = {'C': [[1, (0.6, 1)]]}
        cell.unit = 'B'
        cell.build(0,0)

        ao2 = ft_ao.ft_aopair(cell, cell.Gv)
        nao = cell.nao_nr()
        coords = cell.get_uniform_grids()
        aoR = cell.pbc_eval_gto('GTOval', coords)
        aoR2 = numpy.einsum('ki,kj->kij', aoR.conj(), aoR)
        ngrids = aoR.shape[0]

        ao2ref = [tools.fft(aoR2[:,i,j], cell.mesh) * cell.vol/ngrids
                  for i in range(nao) for j in range(nao)]
        ao2ref = numpy.array(ao2ref).reshape(6,6,-1).transpose(2,0,1)
        self.assertAlmostEqual(abs(ao2ref - ao2).max(), 0, 8)

        aoG = ft_ao.ft_ao(cell, cell.Gv)
        aoref = [tools.fft(aoR[:,i], cell.mesh) * cell.vol/ngrids
                 for i in range(nao)]
        self.assertAlmostEqual(abs(numpy.array(aoref).T - aoG).max(), 0, 8)

    def test_ft_aopair_bvk(self):
        from pyscf.pbc.tools import k2gamma
        n = 2
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.mesh = numpy.array([n,n,n])
        cell.atom = '''C    1.3    .2       .3
                       C     .1    .1      1.1
                       '''
        cell.basis = 'ccpvdz'
        cell.unit = 'B'
        cell.build()

        kpts = cell.make_kpts([2,2,2])
        Gv, Gvbase, kws = cell.get_Gv_weights()
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

        ref = ft_ao.ft_aopair_kpts(cell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase, kptjs=kpts)
        aopair = ft_ao.ft_aopair_kpts(cell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase,
                                      kptjs=kpts, bvk_kmesh=bvk_kmesh)
        #FIXME: error seems too big
        self.assertAlmostEqual(abs(ref - aopair).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(aopair), (-5.735639500461687-12.425151458809875j), 6)

if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
