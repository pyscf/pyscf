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
from pyscf import gto, dft, lib
from pyscf.dft import radi
from pyscf.grad import rks
try:
    from pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None


def grids_response(grids):
    # JCP 98, 5612 (1993); DOI:10.1063/1.464906
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.mole.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = numpy.empty((mol.natm,ngrids))
        for ia in range(mol.natm):
            dc = coords - atm_coords[ia]
            grid_dist[ia] = numpy.linalg.norm(dc,axis=1) + 1e-200

        pbecke = numpy.ones((mol.natm,ngrids))
        for i in range(mol.natm):
            for j in range(i):
                g = 1/atm_dist[i,j] * (grid_dist[i]-grid_dist[j])
                g = fadjust(i, j, g)
                g = (3 - g**2) * g * .5
                g = (3 - g**2) * g * .5
                g = (3 - g**2) * g * .5
                pbecke[i] *= .5 * (1-g + 1e-200)
                pbecke[j] *= .5 * (1+g + 1e-200)

        dpbecke = numpy.zeros((mol.natm,mol.natm,ngrids,3))
        for ia in range(mol.natm):
            for ib in range(mol.natm):
                if ib != ia:
                    g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                    p0 = gadjust(ia, ib, g)
                    g = fadjust(ia, ib, g)
                    p1 = (3 - g **2) * g  * .5
                    p2 = (3 - p1**2) * p1 * .5
                    p3 = (3 - p2**2) * p2 * .5
                    s_uab = .5 * (1 - p3 + 1e-200)
                    t_uab = -27./16 * (1-p2**2) * (1-p1**2) * (1-g**2)
                    t_uab /= s_uab
                    t_uab *= p0

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                    uab = atm_coords[ia] - atm_coords[ib]
                    if ia == atom_id:  # dA PA: dA~ib, PA~ia
                        ua = atm_coords[ib] - coords
                        d_uab = ua/grid_dist[ib,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ia]-grid_dist[ib])/atm_dist[ia,ib]**3
                        d_uab-= v[:,None] * uab
                        dpbecke[ia,ia] += (pbecke[ia]*t_uab).reshape(-1,1) * d_uab
                    else:  # dB PB: dB~ib, PB~ia
                        ua = atm_coords[ia] - coords
                        d_uab = ua/grid_dist[ia,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ia]-grid_dist[ib])/atm_dist[ia,ib]**3
                        d_uab-= v[:,None] * uab
                        dpbecke[ia,ia] += (pbecke[ia]*t_uab).reshape(-1,1) * d_uab

                        if ib != atom_id:  # dA PB: dA~atom_id PB~ia D~ib
                            ua_ub = ((coords-atm_coords[ia])/grid_dist[ia,:,None] -
                                     (coords-atm_coords[ib])/grid_dist[ib,:,None])
                            ua_ub /= atm_dist[ia,ib]
                            dpbecke[atom_id,ia] += (pbecke[ia]*t_uab)[:,None] * ua_ub

                    uba = atm_coords[ib] - atm_coords[ia]
                    if ib == atom_id:  # dA PB: dA~ib PB~ia
                        ub = atm_coords[ia] - coords
                        d_uba = ub/grid_dist[ia,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ib]-grid_dist[ia])/atm_dist[ia,ib]**3
                        d_uba-= v[:,None] * uba
                        dpbecke[ib,ia] += -(pbecke[ia]*t_uab).reshape(-1,1) * d_uba
                    else:  # dB PC: dB~ib, PC~ia and dB PA: dB~ib, PA~ia
                        ub = atm_coords[ib] - coords
                        d_uba = ub/grid_dist[ib,:,None]/atm_dist[ia,ib]
                        v = (grid_dist[ib]-grid_dist[ia])/atm_dist[ia,ib]**3
                        d_uba-= v[:,None] * uba
                        dpbecke[ib,ia] += -(pbecke[ia]*t_uab).reshape(-1,1) * d_uba
        return pbecke, dpbecke

    ngrids = 0
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        ngrids += vol.size

    coords_all = numpy.zeros((ngrids,3))
    w0 = numpy.zeros((ngrids))
    w1 = numpy.zeros((mol.natm,ngrids,3))
    p1 = 0
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords = coords + atm_coords[ia]
        p0, p1 = p1, p1 + vol.size
        coords_all[p0:p1] = coords
        pbecke, dpbecke = gen_grid_partition(coords, ia)
        z = pbecke.sum(axis=0)
        for ib in range(mol.natm):  # derivative wrt to atom_ib
            dz = dpbecke[ib].sum(axis=0)
            w1[ib,p0:p1] = dpbecke[ib,ia]/z[:,None] - (pbecke[ia]/z**2)[:,None]*dz
            w1[ib,p0:p1] *= vol[:,None]

        w0[p0:p1] = vol * pbecke[ia] / z
    return coords_all, w0, w1

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-14

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
        mf.kernel()

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_finite_diff_rks_grad(self):
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.049887866191414401, 5)

# O     0.0000000000    -0.0000000000     0.0210225191
# H     0.0000000000     0.0281984036    -0.0105112595
# H    -0.0000000000    -0.0281984036    -0.0105112595
        g = mf.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.049891265876709084, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_rks_d3_grad(self):
        mol1 = mol.copy()
        mf = dft.RKS(mol, xc='b3lyp')
        mf.conv_tol = 1e-14
        mf.kernel()
        g = mf.nuc_grad_method().set(grid_response=True).kernel()

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_rks_d4_grad(self):
        mol1 = mol.copy()
        mf = dft.RKS(mol, xc='b3lyp')
        mf.conv_tol = 1e-14
        mf.kernel()
        g = mf.nuc_grad_method().set(grid_response=True).kernel()

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_df_rks_grad(self):
        mf1 = mf.density_fit ().run ()
        g = mf1.nuc_grad_method ().set (grid_response=True).kernel ()
        self.assertAlmostEqual(lib.fp(g), -0.04990623577718451, 5)

        mol1 = mol.copy()
        mf_scanner = mf1.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_df_rks_d3_grad(self):
        mf1 = dft.RKS(mol, xc='b3lyp').density_fit ()
        mf1.disp = 'd3bj'
        mf1.kernel()
        g = mf1.nuc_grad_method ().set (grid_response=True).kernel ()

        mol1 = mol.copy()
        mf_scanner = mf1.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_consistency_df_rks_d3_grad(self):
        mf1 = dft.RKS(mol, xc='b3lyp').density_fit ()
        mf1.disp = 'd3bj'
        mf1.kernel()
        g1 = mf1.nuc_grad_method ().set (grid_response=True).kernel ()

        mf2 = dft.RKS(mol, xc='b3lyp-d3bj').density_fit ()
        mf2.kernel()
        g2 = mf2.nuc_grad_method ().set (grid_response=True).kernel ()
        self.assertAlmostEqual(lib.fp(g1), lib.fp(g2), 5)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_df_rks_d4_grad(self):
        mf1 = dft.RKS(mol, xc='b3lyp').density_fit ()
        mf1.disp = 'd4'
        mf1.kernel()
        g = mf1.nuc_grad_method ().set (grid_response=True).kernel ()

        mol1 = mol.copy()
        mf_scanner = mf1.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_rks_grad_lda(self):
        mol_hf = gto.Mole()
        mol_hf.atom = [
            ['H' , (0. , 0. , 1.804)],
            ['F' , (0. , 0. , 0.   )], ]
        mol_hf.unit = 'B'
        mol_hf.basis = '631g'
        mol_hf.build()

#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
        mf = dft.RKS(mol_hf).run(conv_tol=1e-14)
        self.assertAlmostEqual(lib.fp(mf.Gradients().kernel()), 0.0018831588319051444, 6)

#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]
        mf.grids.prune = None
        mf.grids.level = 6
        mf.run(conv_tol=1e-14)
        self.assertAlmostEqual(lib.fp(mf.Gradients().kernel()), 0.0018819497229394144, 6)

    def test_finite_diff_rks_grad_gga(self):
#[[ -3.59411142e-16  -2.68753987e-16   1.21557501e-02]
# [  4.04977877e-17   2.11112794e-02  -6.08181640e-03]
# [  1.52600378e-16  -2.11112794e-02  -6.08181640e-03]]
        mf = mol.RKS().run(xc='b3lypg', conv_tol=1e-12)
        g = mf.nuc_grad_method().set().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.035613964330885352, 5)

#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
        mf = mol.RKS().run(xc='b88,p86', conv_tol=1e-12)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.050382923259300716, 5)

        g = mf.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.05036316927480719, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-3*lib.param.BOHR, 5)

    def test_finite_diff_rks_grad_nlc(self):
#[[ 2.91036539e-16  1.22693574e-15  2.45978284e-02]
# [ 2.83888198e-17  2.66388957e-02 -1.23039325e-02]
# [ 1.17327811e-16 -2.66388957e-02 -1.23039325e-02]]
        mf = mol.RKS()
        mf.set(xc='VV10', nlc='VV10', conv_tol=1e-12)
        mf.nlcgrids.level = 1
        mf.kernel()
        g = mf.nuc_grad_method().set().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.049431714073528615, 5)

        mf.nlcgrids.level = 0
        mf.kernel()
        g = mf.nuc_grad_method().set(grid_response=True).kernel()

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-3*lib.param.BOHR, 5)

        mf.set(xc='wb97m-v', nlc='', conv_tol=1e-12)
        mf.kernel()
        g = mf.nuc_grad_method().set().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.032453247110454116, 5)

        mf.set(xc='wb97m-v', nlc=False, conv_tol=1e-12)
        mf.kernel()
        g = mf.nuc_grad_method().set().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.03202508099735074, 5)

    def test_finite_diff_rks_grad_mgga(self):
        mf = mol.RKS().run(xc='m06l', conv_tol=1e-12)
        g = mf.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.02804214421452781, 6)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-3*lib.param.BOHR, 5)

    def test_different_grids_for_grad(self):
        grids1 = dft.gen_grid.Grids(mol)
        grids1.level = 1
        g = mf.nuc_grad_method().set(grids=grids1).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.049837230292484727, 6)

    def test_grid_response(self):
        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = '''
            H   0.   0  -0.50001
            C   0.   1    .1
            O   0.   0   0.5
            F   1.   .3  0.5'''
        mol1.unit = 'B'
        mol1.build()
        grids1 = dft.gen_grid.Grids(mol1)
        c, w0b, w1b = grids_response(grids1)

        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = '''
            H   0.   0  -0.5
            C   0.   1    .1
            O   0.   0   0.5
            F   1.   .3  0.5'''
        mol.unit = 'B'
        mol.build()
        grids = dft.gen_grid.Grids(mol)
        c, w0a, w1a = grids_response(grids)
        self.assertAlmostEqual(lib.fp(w1a.transpose(0,2,1)), -13.101186585274547, 10)

        mol0 = gto.Mole()
        mol0.verbose = 0
        mol0.atom = '''
            H   0.   0  -0.49999
            C   0.   1    .1
            O   0.   0   0.5
            F   1.   .3  0.5'''
        mol0.unit = 'B'
        mol0.build()
        grids0 = dft.gen_grid.Grids(mol0)
        c, w0a      = grids_response(grids0)[:2]
        dw = (w0a-w0b) / .00002
        self.assertTrue(abs(dw-w1a[0,:,2]).max() < 1e-5)

        coords = []
        w0 = []
        w1 = []
        for c_a, w0_a, w1_a in rks.grids_response_cc(grids):
            coords.append(c_a)
            w0.append(w0_a)
            w1.append(w1_a)
        coords = numpy.vstack(coords)
        w0 = numpy.hstack(w0)
        w1 = numpy.concatenate(w1, axis=2)
        self.assertAlmostEqual(lib.fp(w1), -13.101186585274547, 10)
        self.assertAlmostEqual(abs(w1-w1a.transpose(0,2,1)).max(), 0, 12)

        grids.radii_adjust = radi.becke_atomic_radii_adjust
        coords = []
        w0 = []
        w1 = []
        for c_a, w0_a, w1_a in rks.grids_response_cc(grids):
            coords.append(c_a)
            w0.append(w0_a)
            w1.append(w1_a)
        coords = numpy.vstack(coords)
        w0 = numpy.hstack(w0)
        w1 = numpy.concatenate(w1, axis=2)
        self.assertAlmostEqual(lib.fp(w1), -163.85086096365865, 9)


    def test_get_vxc(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.build()
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-12
        mf.grids.radii_adjust = radi.becke_atomic_radii_adjust
        mf.scf()
        g = rks.Gradients(mf)
        g.grid_response = True
        g0 = g.kernel()
        dm0 = mf.make_rdm1()

        mol0 = gto.Mole()
        mol0.verbose = 0
        mol0.atom = [
            ['O' , (0. , 0.     ,-0.00001)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol0.basis = '631g'
        mol0.build()
        mf0 = dft.RKS(mol0)
        mf0.grids.radii_adjust = radi.becke_atomic_radii_adjust
        mf0.conv_tol = 1e-12
        e0 = mf0.scf()

        denom = 1/.00002 * lib.param.BOHR
        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = [
            ['O' , (0. , 0.     , 0.00001)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol1.basis = '631g'
        mol1.build()
        mf1 = dft.RKS(mol1)
        mf1.grids.radii_adjust = radi.becke_atomic_radii_adjust
        mf1.conv_tol = 1e-12
        e1 = mf1.scf()
        self.assertAlmostEqual((e1-e0)*denom, g0[0,2], 6)

        # grids response have non-negligible effects for small grids
        grids = dft.gen_grid.Grids(mol)
        grids.atom_grid = (20,86)
        grids.build(with_non0tab=False)
        grids0 = dft.gen_grid.Grids(mol0)
        grids0.atom_grid = (20,86)
        grids0.build(with_non0tab=False)
        grids1 = dft.gen_grid.Grids(mol1)
        grids1.atom_grid = (20,86)
        grids1.build(with_non0tab=False)
        xc = 'lda,'
        exc0 = dft.numint.nr_rks(mf0._numint, mol0, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_rks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_rks(mf0._numint, mol0, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_rks(mf1._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = rks.get_vxc(mf._numint, mol, grids, xc, dm0)[1]
        ev1, vxc1 = rks.get_vxc_full_response(mf._numint, mol, grids, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm0[p0:p1])*2
        exc1_full = numpy.einsum('xij,ij->x', vxc1[:,p0:p1], dm0[p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 3)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 7)

        xc = 'pbe,'
        exc0 = dft.numint.nr_rks(mf0._numint, mol0, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_rks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_rks(mf0._numint, mol0, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_rks(mf1._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = rks.get_vxc(mf._numint, mol, grids, xc, dm0)[1]
        ev1, vxc1 = rks.get_vxc_full_response(mf._numint, mol, grids, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm0[p0:p1])*2
        exc1_full = numpy.einsum('xij,ij->x', vxc1[:,p0:p1], dm0[p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 2)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 7)

        xc = 'pbe0'
        grids.radii_adjust = None
        grids0.radii_adjust = None
        grids1.radii_adjust = None
        exc0 = dft.numint.nr_rks(mf0._numint, mol0, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_rks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_rks(mf0._numint, mol0, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_rks(mf1._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = rks.get_vxc(mf._numint, mol, grids, xc, dm0)[1]
        ev1, vxc1 = rks.get_vxc_full_response(mf._numint, mol, grids, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm0[p0:p1])*2
        exc1_full = numpy.einsum('xij,ij->x', vxc1[:,p0:p1], dm0[p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 1)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 7)

        xc = 'm06l,'
        exc0 = dft.numint.nr_rks(mf0._numint, mol0, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_rks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_rks(mf0._numint, mol0, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_rks(mf1._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = rks.get_vxc(mf._numint, mol, grids, xc, dm0)[1]
        ev1, vxc1 = rks.get_vxc_full_response(mf._numint, mol, grids, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('xij,ij->x', vxc[:,p0:p1], dm0[p0:p1])*2
        exc1_full = numpy.einsum('xij,ij->x', vxc1[:,p0:p1], dm0[p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 1)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 7)

    def test_range_separated(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'wb97x'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -76.36324548001258, 8)

        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.0270988199051937, 5)

        mol1 = gto.M(atom="H; H 1 1.", basis='ccpvdz', verbose=0)
        mf = dft.RKS(mol1)
        mf.xc = 'wb97x'
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.17166479488374434, 5)

        smf = mf.as_scanner()
        mol1 = gto.M(atom="H; H 1 1.001", basis='ccpvdz')
        mol2 = gto.M(atom="H; H 1 0.999", basis='ccpvdz')
        dx = (mol1.atom_coord(1) - mol2.atom_coord(1))[0]
        e1 = smf(mol1)
        e2 = smf(mol2)
        self.assertAlmostEqual((e1-e2)/dx, g[1,0], 5)


if __name__ == "__main__":
    print("Full Tests for RKS Gradients")
    unittest.main()
