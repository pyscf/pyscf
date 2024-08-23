#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import numpy as np

from pyscf import gto, lib
from pyscf.dft import rks
import pyscf.dft

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
import pyscf.pbc
pyscf.pbc.DEBUG = False


def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    gamma_point = kpt is None or abs(kpt).sum() < 1e-9
    aoR = 0
    for L in cell.get_lattice_Ls():
        if gamma_point:
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, deriv,
                                            shls_slice, non0tab, out, verbose)
        else:
            factor = numpy.exp(1j*numpy.dot(kpt,L))
            aoR += pyscf.dft.numint.eval_ao(cell, coords-L, deriv,
                                            shls_slice, non0tab, out, verbose) * factor
    return numpy.asarray(aoR)

def coords_wrap_around_phase(cell, coords, kpts):
    ngrids = coords.shape[0]
    fac = np.exp(1j*cell.lattice_vectors().dot(kpts.T))
    fac = np.repeat(fac[None,:,:], ngrids, axis=0)
    fac[coords>=0] = 1
    fac = np.prod(fac, axis=1)
    return fac

def make_grids(mesh):
    L = 60
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.unit = 'B'
    cell.a = ((L,0,0),(0,L,0),(0,0,L))
    cell.mesh = mesh

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = {'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.build(False, False)
    grids = gen_grid.UniformGrids(cell)
    grids.build()
    return cell, grids


class KnownValues(unittest.TestCase):
    def test_eval_ao(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.a = np.eye(3) * 2.5
        cell.mesh = [21]*3
        cell.atom = [['C', (1., .8, 1.9)],
                     ['C', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.precision = 1e-11
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        ni = numint.NumInt()
        ao10 = eval_ao(cell, grids.coords, deriv=1)
        ao0 = ao10[0]
        ao1 = ni.eval_ao(cell, grids.coords)
        self.assertAlmostEqual(abs(ao0-ao1).max(), 0, 11)
        self.assertAlmostEqual(lib.fp(ao1), -0.54069672246407219, 8)

        ao11 = ni.eval_ao(cell, grids.coords, deriv=1)
        self.assertAlmostEqual(abs(ao10-ao11).max(), 0, 10)
        self.assertAlmostEqual(lib.fp(ao11), 8.8004405892746433, 8)

        ni.non0tab = ni.make_mask(cell, grids.coords)
        ao1 = ni.eval_ao(cell, grids.coords)
        self.assertAlmostEqual(abs(ao0-ao1).max(), 0, 10)
        self.assertAlmostEqual(lib.fp(ao1), -0.54069672246407219, 8)

        ao11 = ni.eval_ao(cell, grids.coords, deriv=1)
        self.assertAlmostEqual(abs(ao10-ao11).max(), 0, 10)
        self.assertAlmostEqual(lib.fp(ao11), 8.8004405892746433, 8)

        ao11 = ni.eval_ao(cell, grids.coords, deriv=1, shls_slice=(3,7))
        self.assertAlmostEqual(abs(ao10[:,:,6:17] - ao11).max(), 0, 10)


    def test_eval_mat(self):
        cell, grids = make_grids([61]*3)
        ng = grids.weights.size
        np.random.seed(1)
        rho = np.random.random(ng)
        rho *= 1/np.linalg.norm(rho)
        vrho = np.random.random(ng)
        ao1 = numint.eval_ao(cell, grids.coords)
        mat1 = numint.eval_mat(cell, ao1, grids.weights, rho, vrho)
        w = np.arange(mat1.size) * .01
        self.assertAlmostEqual(np.dot(w,mat1.ravel()), (.14777107967912118+0j), 7)

    def test_eval_ao_kpts(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.a = np.eye(3) * 2.5
        cell.mesh = [21]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.precision = 1e-11
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        np.random.seed(1)
        kpts = np.random.random((4,3))
        ni = numint.KNumInt(kpts)
        coords = grids.coords
        ao1 = ni.eval_ao(cell, coords, kpts)
        fac = coords_wrap_around_phase(cell, coords, kpts)

        self.assertAlmostEqual(lib.fp(ao1[0]*fac[:,0:1]), (-2.4066959390326477-0.98044994099240701j), 8)
        self.assertAlmostEqual(lib.fp(ao1[1]*fac[:,1:2]), (-0.30643153325360639+0.1571658820483913j), 8)
        self.assertAlmostEqual(lib.fp(ao1[2]*fac[:,2:3]), (-1.1937974302337684-0.39039259235266233j), 8)
        self.assertAlmostEqual(lib.fp(ao1[3]*fac[:,3:4]), (0.17701966968272009-0.20232879692603079j), 8)

        coords = cell.get_uniform_grids(wrap_around=False)
        ao1 = ni.eval_ao(cell, coords, kpts)
        self.assertAlmostEqual(lib.fp(ao1[0]), (-2.4066959390326477-0.98044994099240701j), 8)
        self.assertAlmostEqual(lib.fp(ao1[1]), (-0.30643153325360639+0.1571658820483913j), 8)
        self.assertAlmostEqual(lib.fp(ao1[2]), (-1.1937974302337684-0.39039259235266233j), 8)
        self.assertAlmostEqual(lib.fp(ao1[3]), (0.17701966968272009-0.20232879692603079j), 8)

    def test_eval_ao_kpt(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.a = np.eye(3) * 2.5
        cell.mesh = [21]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.precision = 1e-11
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        np.random.seed(1)
        kpt = np.random.random(3)
        ni = numint.NumInt()
        ao0 = eval_ao(cell, grids.coords, kpt)
        ao1 = ni.eval_ao(cell, grids.coords, kpt)
        self.assertAlmostEqual(abs(ao0 - ao1).max(), 0, 10)
        fac = coords_wrap_around_phase(cell, grids.coords, kpt[None])
        self.assertAlmostEqual(lib.fp(ao1*fac), (-2.4066959390326477-0.98044994099240701j), 8)

    def test_nr_rks(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.a = np.eye(3) * 2.5
        cell.mesh = [21]*3
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = 'ccpvdz'
        cell.precision = 1e-11
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()
        nao = cell.nao_nr()

        np.random.seed(1)
        kpts = np.random.random((2,3))
        dms = np.random.random((2,nao,nao))
        dms = (dms + dms.transpose(0,2,1)) * .5
        ni = numint.NumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms[0], hermi=1, kpt=kpts[0])
        self.assertAlmostEqual(ne, 5.0499199224525153, 8)
        self.assertAlmostEqual(exc, -3.8870579114663886, 8)
        self.assertAlmostEqual(lib.fp(vmat), 0.42538491159934377+0.14139753327162483j, 8)

        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms[0], hermi=0, kpt=kpts[0])
        self.assertAlmostEqual(lib.fp(vmat), 0.42538491159934377+0.14139753327162483j, 8)

        ni = numint.KNumInt()
        with lib.temporary_env(pbcgto.eval_gto, EXTRA_PREC=1e-5):
            ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms, hermi=1, kpts=kpts)
        self.assertAlmostEqual(ne, 6.0923292346269742, 8)
        self.assertAlmostEqual(exc, -3.9899423803106466, 8)
        self.assertAlmostEqual(lib.fp(vmat[0]), -2348.9577179701278-60.733087913116719j, 5)
        self.assertAlmostEqual(lib.fp(vmat[1]), -2353.0350086740673-117.74811536967495j, 5)

        with lib.temporary_env(pbcgto.eval_gto, EXTRA_PREC=1e-5):
            ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', [dms,dms], hermi=1, kpts=kpts)
        self.assertAlmostEqual(ne[1], 6.0923292346269742, 8)
        self.assertAlmostEqual(exc[1], -3.9899423803106466, 8)
        self.assertAlmostEqual(lib.fp(vmat[1][0]), -2348.9577179701278-60.733087913116719j, 5)
        self.assertAlmostEqual(lib.fp(vmat[1][1]), -2353.0350086740673-117.74811536967495j, 5)

        cell = pbcgto.Cell()
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-pbe'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()
        nao = cell.nao
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        np.random.seed(1)
        kpts = np.random.random((2,3))
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        ne, exc, vmat1 = ni.nr_rks(cell, grids, 'blyp', dm, hermi=1, kpt=kpts[0])
        ne, exc, vmat2 = ni.nr_rks(cell, grids, 'blyp', dm, hermi=0, kpt=kpts[0])
        self.assertAlmostEqual(lib.fp(vmat1), (6.238900947350686+0.6026114431281391j), 8)
        self.assertAlmostEqual(abs(vmat1 - vmat2).max(), 0, 9)

    def test_eval_rho(self):
        cell, grids = make_grids([61]*3)
        numpy.random.seed(10)
        nao = 10
        ngrids = 500
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ao =(numpy.random.random((10,ngrids,nao)) +
             numpy.random.random((10,ngrids,nao))*1j)
        ao = ao.transpose(0,2,1).copy().transpose(0,2,1)

        rho0 = numpy.zeros((6,ngrids), dtype=numpy.complex128)
        rho0[0] = numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[0].conj())
        rho0[1] = numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[1].conj())
        rho0[2] = numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[2].conj())
        rho0[3] = numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[0].conj()) + numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[3].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[4].conj()) + numpy.einsum('pi,ij,pj->p', ao[4], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[7].conj()) + numpy.einsum('pi,ij,pj->p', ao[7], dm, ao[0].conj())
        rho0[4]+= numpy.einsum('pi,ij,pj->p', ao[0], dm, ao[9].conj()) + numpy.einsum('pi,ij,pj->p', ao[9], dm, ao[0].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[1], dm, ao[1].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[2], dm, ao[2].conj())
        rho0[5]+= numpy.einsum('pi,ij,pj->p', ao[3], dm, ao[3].conj())
        rho0[4]+= rho0[5]*2
        rho0[5] *= .5

        rho1 = numint.eval_rho(cell, ao, dm, xctype='MGGA')
        self.assertAlmostEqual(abs(rho0 - rho1).max(), 0, 12)

        rho1 = numint.eval_rho(cell, ao, dm, xctype='GGA')
        self.assertAlmostEqual(lib.fp(rho1), -255.45150185669198, 7)

        rho1 = numint.eval_rho(cell, ao[0], dm, xctype='LDA')
        self.assertAlmostEqual(lib.fp(rho1), -17.198879910245601, 7)

    def test_eval_mat1(self):
        cell, grids = make_grids([61]*3)
        numpy.random.seed(10)
        nao = 10
        ngrids = 500
        rho = numpy.random.random((6,ngrids))
        vxc = numpy.random.random((4,ngrids))
        weight = numpy.random.random(ngrids)
        ao =(numpy.random.random((10,ngrids,nao)) +
             numpy.random.random((10,ngrids,nao))*1j)

        mat0 = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vxc[0], ao[0])
        mat1 = numint.eval_mat(cell, ao[0], weight, rho, vxc, xctype='LDA')
        self.assertAlmostEqual(abs(mat0 - mat1).max(), 0, 12)

        vrho, vsigma = vxc[:2]
        wv = weight * vsigma * 2
        mat0  = numpy.einsum('pi,p,pj->ij', ao[0].conj(), weight*vrho, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[1]*wv, ao[1]) + numpy.einsum('pi,p,pj->ij', ao[1].conj(), rho[1]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[2]*wv, ao[2]) + numpy.einsum('pi,p,pj->ij', ao[2].conj(), rho[2]*wv, ao[0])
        mat0 += numpy.einsum('pi,p,pj->ij', ao[0].conj(), rho[3]*wv, ao[3]) + numpy.einsum('pi,p,pj->ij', ao[3].conj(), rho[3]*wv, ao[0])
        mat1 = numint.eval_mat(cell, ao, weight, rho, vxc, xctype='GGA')
        self.assertAlmostEqual(abs(mat0 - mat1).max(), 0, 11)

        mat1 = numint.eval_mat(cell, ao, weight, rho, vxc, xctype='MGGA')
        self.assertAlmostEqual(lib.fp(mat1), -160.191390949408+21.478570186344374j, 7)

        mat1 = numint.eval_mat(cell, ao[0], weight, rho, vxc, xctype='LDA')
        self.assertAlmostEqual(lib.fp(mat1), 10.483493302918024+3.5590312220458227j, 7)

    def test_2d_rho(self):
        cell = pbcgto.Cell()
        cell.a = '5 0 0; 0 5 0; 0 0 1'
        cell.unit = 'B'
        cell.atom = 'He     1.    0.       1.'
        cell.basis = {'He': '321g'}
        cell.dimension = 2
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.verbose = 0
        cell.mesh = [10,10,30]
        cell.build()
        grids = gen_grid.UniformGrids(cell)
        grids.build()
        grids.coords = cell.get_uniform_grids(wrap_around=False)
        numpy.random.seed(10)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        rho = numint.get_rho(ni, cell, dm, grids)
        self.assertAlmostEqual(lib.fp(rho), 7.2089907050590334, 9)

    def test_1d_rho(self):
        cell = pbcgto.Cell()
        cell.a = '5 0 0; 0 1 0; 0 0 1'
        cell.unit = 'B'
        cell.atom = 'He     1.    0.       1.'
        cell.basis = {'He': '321g'}
        cell.dimension = 1
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.verbose = 0
        cell.mesh = [10,30,30]
        cell.build()
        grids = gen_grid.UniformGrids(cell)
        grids.build()
        grids.coords = cell.get_uniform_grids(wrap_around=False)
        numpy.random.seed(10)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        rho = numint.get_rho(ni, cell, dm, grids)
        self.assertAlmostEqual(lib.fp(rho), 1.1624587519868457, 9)

    def test_3d_rho(self):
        cell = pbcgto.Cell()
        cell.a = '3 0 0; 0 3 0; 0 0 3'
        cell.unit = 'B'
        cell.atom = 'He     1.    0.       1.'
        cell.basis = {'He': '321g'}
        cell.verbose = 0
        cell.mesh = [20,20,20]
        cell.build()
        grids = gen_grid.UniformGrids(cell)
        grids.build()
        numpy.random.seed(10)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        rho = numint.get_rho(ni, cell, dm, grids)
        self.assertAlmostEqual(lib.fp(rho), 1.4639787098513968, 8)

        grids.coords = cell.get_uniform_grids(wrap_around=False)
        rho1 = numint.get_rho(ni, cell, dm, grids)
        self.assertAlmostEqual(abs(rho - rho1).max(), 0, 8)

    def test_nr_uks_vxc_vv10(self):
        cell = pbcgto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.a = np.eye(3) * 2.5
        cell.atom = [['He', (1., .8, 1.9)],
                     ['He', (.1, .2,  .3)],]
        cell.basis = [[0, [1, 1]], [1, [.5, 1]]]
        cell.build(False, False)
        grids = gen_grid.UniformGrids(cell)
        grids.build()

        # TODO: a better test case to show more differences between RKS-nlc and KRKS-nlc
        dm = cell.RKS().get_init_guess()
        ni = numint.NumInt()
        n, e, v = numint.nr_nlc_vxc(ni, cell, grids, 'wB97M_V', dm)
        self.assertAlmostEqual(n, 4, 7)
        self.assertAlmostEqual(e, 0.0184789363, 7)
        self.assertAlmostEqual(lib.fp(v), 0.0026492515, 7)

        kpts = cell.make_kpts([3,1,1])
        dm = cell.KRKS(kpts=kpts).get_init_guess()
        ni = numint.KNumInt()
        n, e, v = numint.nr_nlc_vxc(ni, cell, grids, 'wB97M_V', dm, kpts=kpts)
        self.assertAlmostEqual(n, 4, 7)
        self.assertAlmostEqual(e, 0.0184756407, 7)
        self.assertAlmostEqual(lib.fp(v), 0.0026498869, 7)

        h2o = pbcgto.Cell()
        h2o.a = np.eye(3) * 10
        h2o.atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        h2o.basis = {"H": '6-31g', "O": '6-31g',}
        h2o.build()

        method = h2o.UKS().density_fit()
        dm = method.get_init_guess()
        grids = gen_grid.BeckeGrids(h2o)
        grids.atom_grid = {'H': (20, 50), 'O': (20,50)}
        grids.build()
        ni = numint.NumInt()
        n, e, v = numint.nr_nlc_vxc(ni, h2o, grids, 'wB97M_V', dm[0]*2, hermi=0)
        self.assertAlmostEqual(e, 0.042436, 4)
        self.assertAlmostEqual(lib.fp([v, v]), 0.0229223, 4)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()
