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

import unittest
import numpy
import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pgto
import pyscf.pbc.dft as pdft
from pyscf.pbc.df import fft, aft, mdf, rsdf_builder, gdf_builder




##################################################
#
# port from ao2mo/eris.py
#
##################################################
from pyscf.pbc import lib as pbclib
from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc import tools

einsum = np.einsum

r"""
    (ij|kl) = \int dr1 dr2 i*(r1) j(r1) v(r12) k*(r2) l(r2)
            = (ij|G) v(G) (G|kl)

    i*(r) j(r) = 1/N \sum_G e^{iGr}  (G|ij)
               = 1/N \sum_G e^{-iGr} (ij|G)

    "forward" FFT:
        (G|ij) = \sum_r e^{-iGr} i*(r) j(r) = fft[ i*(r) j(r) ]
    "inverse" FFT:
        (ij|G) = \sum_r e^{iGr} i*(r) j(r) = N * ifft[ i*(r) j(r) ]
               = conj[ \sum_r e^{-iGr} j*(r) i(r) ]
"""

def general(cell, mo_coeffs, kpts=None, compact=0):
    '''pyscf-style wrapper to get MO 2-el integrals.'''
    assert len(mo_coeffs) == 4
    if kpts is not None:
        assert len(kpts) == 4
    return get_mo_eri(cell, mo_coeffs, kpts)

def get_mo_eri(cell, mo_coeffs, kpts=None):
    '''Convenience function to return MO 2-el integrals.'''
    mo_coeff12 = mo_coeffs[:2]
    mo_coeff34 = mo_coeffs[2:]
    if kpts is None:
        kpts12 = kpts34 = q = None
    else:
        kpts12 = kpts[:2]
        kpts34 = kpts[2:]
        q = kpts12[0] - kpts12[1]
        #q = kpts34[1] - kpts34[0]
    if q is None:
        q = np.zeros(3)

    mo_pairs12_kG = get_mo_pairs_G(cell, mo_coeff12, kpts12)
    mo_pairs34_invkG = get_mo_pairs_invG(cell, mo_coeff34, kpts34, q)
    return assemble_eri(cell, mo_pairs12_kG, mo_pairs34_invkG, q)

def get_mo_pairs_G(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate forward (G|ij) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G : (ngrids, nmoi*nmoj) ndarray
            The FFT of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(-1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R_ij, cell.mesh, fac)

    return mo_pairs_G

def get_mo_pairs_invG(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_invG : (ngrids, nmoi*nmoj) ndarray
            The inverse FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_invG = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R_ij), cell.mesh, fac))

    return mo_pairs_invG

def get_mo_pairs_G_old(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G, mo_pairs_invG : (ngrids, nmoi*nmoj) ndarray
            The FFTs of the real-space MO pairs.
    '''
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    mo_pairs_R = np.einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngrids,nmoi*nmoj], np.complex128)
    mo_pairs_invG = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(-1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R[:,i,j], cell.mesh, fac)
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R[:,i,j]), cell.mesh,
                                                                   fac.conj()))

    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None):
    '''Assemble 4-index electron repulsion integrals.

    Returns:
        (nmo1*nmo2, nmo3*nmo4) ndarray

    '''
    if q is None:
        q = np.zeros(3)

    coulqG = tools.get_coulG(cell, -1.0*q)
    ngrids = orb_pair_invG1.shape[0]
    Jorb_pair_G2 = np.einsum('g,gn->gn',coulqG,orb_pair_G2)*(cell.vol/ngrids**2)
    eri = np.dot(orb_pair_invG1.T, Jorb_pair_G2)
    return eri

def get_ao_pairs_G(cell, kpt=np.zeros(3)):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngrids, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords, kpt) # shape = (coords, nao)
    ngrids, nao = aoR.shape
    gamma_point = abs(kpt).sum() < 1e-9
    if gamma_point:
        npair = nao*(nao+1)//2
        ao_pairs_G = np.empty([ngrids, npair], np.complex128)

        ij = 0
        for i in range(nao):
            for j in range(i+1):
                ao_ij_R = np.conj(aoR[:,i]) * aoR[:,j]
                ao_pairs_G[:,ij] = tools.fft(ao_ij_R, cell.mesh)
                #ao_pairs_invG[:,ij] = ngrids*tools.ifft(ao_ij_R, cell.mesh)
                ij += 1
        ao_pairs_invG = ao_pairs_G.conj()
    else:
        ao_pairs_G = np.zeros([ngrids, nao,nao], np.complex128)
        for i in range(nao):
            for j in range(nao):
                ao_ij_R = np.conj(aoR[:,i]) * aoR[:,j]
                ao_pairs_G[:,i,j] = tools.fft(ao_ij_R, cell.mesh)
        ao_pairs_invG = ao_pairs_G.transpose(0,2,1).conj().reshape(-1,nao**2)
        ao_pairs_G = ao_pairs_G.reshape(-1,nao**2)
    return ao_pairs_G, ao_pairs_invG

def get_ao_eri(cell, kpt=np.zeros(3)):
    '''Convenience function to return AO 2-el integrals.'''

    ao_pairs_G, ao_pairs_invG = get_ao_pairs_G(cell, kpt)
    eri = assemble_eri(cell, ao_pairs_invG, ao_pairs_G)
    if abs(kpt).sum() < 1e-9:
        eri = eri.real
    return eri

##################################################
#
# ao2mo/eris.py end
#
##################################################


def setUpModule():
    global cell, cell1, kdf0, kpts, kpt0
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (1., 1)), (1, (.4, 1))],
                  'C' :[[0, [1., 1]]],}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = np.eye(3) * 2.5
    cell.build()
    np.random.seed(1)
    kpts = np.random.random((4,3))
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    kpt0 = np.zeros(3)

    cell1 = pgto.Cell()
    cell1.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell1.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
    cell1.a = np.eye(3) * 2.5
    cell1.mesh = [21] * 3
    cell1.build()
    kdf0 = mdf.MDF(cell1)
    kdf0.auxbasis = 'weigend'
    kdf0.mesh = [21] * 3
    kdf0.kpts = kpts

def tearDownModule():
    global cell, cell1, kdf0
    del cell, cell1, kdf0

class KnownValues(unittest.TestCase):
    def test_aft_get_pp(self):
        v0 = fft.FFTDF(cell, kpts[0]).get_pp()
        v1 = aft.AFTDF(cell, kpts[0]).get_pp()
        v2 = rsdf_builder._RSNucBuilder(cell, kpts[0]).get_pp()
        v3 = gdf_builder._CCNucBuilder(cell, kpts[0]).get_pp()
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v2).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v3).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v0) - (-1.387194815780133+0.007570691824169233j), 0, 8)

        kpts4 = cell.make_kpts([4,1,1])
        v0 = fft.FFTDF(cell, kpts4).get_pp()
        v1 = aft.AFTDF(cell, kpts4).get_pp()
        v2 = rsdf_builder._RSNucBuilder(cell, kpts4).get_pp()
        v3 = gdf_builder._CCNucBuilder(cell, kpts4).get_pp()
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v2).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v3).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v0) - (-5.7070290795125445-0.00038722541238732697j), 0, 8)

    # issue 2575
    def test_aft_get_pp1(self):
        cell = pgto.M(atom='Cu .0 .0 .0', a=np.eye(3)*3, spin=1,
               basis=[[1, [1, 1]]], pseudo='''Cu
1    0   10
0.53    0
1
0.26    1   1.8''')
        kpts = cell.make_kpts([2,1,1])
        ref = fft.FFTDF(cell, kpts).get_pp()
        v = aft.AFTDF(cell, kpts).get_pp()
        self.assertAlmostEqual(abs(v - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v), 0.37494551685809624, 8)

    def test_aft_get_nuc(self):
        v0 = fft.FFTDF(cell, kpts[0]).get_nuc()
        v1 = aft.AFTDF(cell, kpts[0]).get_nuc()
        v2 = rsdf_builder._RSNucBuilder(cell, kpts[0]).get_nuc()
        v3 = gdf_builder._CCNucBuilder(cell, kpts[0]).get_nuc()
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v2).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v3).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v0) - (-3.027291421291857+0.027413088538826163j), 0, 8)

        kpts4 = cell.make_kpts([4,1,1])
        v0 = fft.FFTDF(cell, kpts4).get_nuc()
        v1 = aft.AFTDF(cell, kpts4).get_nuc()
        v2 = rsdf_builder._RSNucBuilder(cell, kpts4).get_nuc()
        v3 = gdf_builder._CCNucBuilder(cell, kpts4).get_nuc()
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v2).max(), 0, 8)
        self.assertAlmostEqual(abs(v0 - v3).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(v0) - (-9.558999451044691-0.0030483827024668946j), 0, 8)

    def test_aft_get_ao_eri(self):
        df0 = fft.FFTDF(cell1)
        df = aft.AFTDF(cell1)
        eri0 = df0.get_ao_eri(compact=True)
        eri1 = df.get_ao_eri(compact=True)
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 8)

        eri0 = df0.get_ao_eri(kpts[0])
        eri1 = df.get_ao_eri(kpts[0])
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 8)

        eri0 = df0.get_ao_eri(kpts)
        eri1 = df.get_ao_eri(kpts)
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 8)

    def test_aft_get_ao_eri_high_cost(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                      'C' :'gth-szv',}
        cell.pseudo = {'C':'gth-pade'}
        cell.a = np.eye(3) * 2.5
        cell.mesh = [21] * 3
        cell.build()
        df0 = fft.FFTDF(cell)
        df = aft.AFTDF(cell)
        eri0 = df0.get_ao_eri(compact=True)
        eri1 = df.get_ao_eri(compact=True)
        self.assertTrue(np.allclose(eri0, eri1, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(lib.fp(eri1), 0.80425361966560172, 8)

        eri0 = df0.get_ao_eri(kpts[0])
        eri1 = df.get_ao_eri(kpts[0])
        self.assertTrue(np.allclose(eri0, eri1, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(lib.fp(eri1), (2.9346374476387949-0.20479054936779137j), 8)

        eri0 = df0.get_ao_eri(kpts)
        eri1 = df.get_ao_eri(kpts)
        self.assertTrue(np.allclose(eri0, eri1, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(lib.fp(eri1), (0.33709287302019619-0.94185725020966538j), 8)

    def test_get_eri_gamma(self):
        odf0 = mdf.MDF(cell1)
        odf0.mesh = [15]* 3
        odf = aft.AFTDF(cell1)
        ref = odf0.get_eri()
        eri0000 = odf.get_eri(compact=True)
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(abs(eri0000-ref).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(eri0000), 0.23714016293926865, 8)

    def test_get_eri_gamma1(self):
        odf = aft.AFTDF(cell1)
        ref = kdf0.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri1111 = odf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2410388899583582-5.2370501878355006e-06j), 8)

        eri1111 = odf.get_eri((kpts[0]+1e-8,kpts[0]+1e-8,kpts[0],kpts[0]))
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2410388899583582-5.2370501878355006e-06j), 8)

    def test_get_eri_0011(self):
        odf = aft.AFTDF(cell1)
        ref = kdf0.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011 = odf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri0011, ref, atol=1e-3, rtol=1e-3))
        self.assertAlmostEqual(lib.fp(eri0011), (1.2410162858084512+0.00074485383749912936j), 8)

        ref = fft.FFTDF(cell1).get_mo_eri([numpy.eye(cell1.nao_nr())]*4, (kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011 = odf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri0011, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0011), (1.2410162860852818+0.00074485383748954838j), 8)

    def test_get_eri_0110(self):
        odf = aft.AFTDF(cell1)
        ref = kdf0.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110 = odf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-6, rtol=1e-6))
        eri0110 = odf.get_eri((kpts[0]+1e-8,kpts[1]+1e-8,kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)

        ref = fft.FFTDF(cell1).get_mo_eri([numpy.eye(cell1.nao_nr())]*4, (kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110 = odf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)
        eri0110 = odf.get_eri((kpts[0]+1e-8,kpts[1]+1e-8,kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)

    def test_get_eri_0123(self):
        odf = aft.AFTDF(cell1)
        ref = kdf0.get_eri(kpts)
        eri1111 = odf.get_eri(kpts)
        self.assertAlmostEqual(abs(eri1111-ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(eri1111), (1.2917759427391706-0.013340252488069412j), 8)

        ref = fft.FFTDF(cell1).get_mo_eri([numpy.eye(cell1.nao_nr())]*4, kpts)
        self.assertAlmostEqual(abs(eri1111-ref).max(), 0, 8)

    def test_get_mo_eri(self):
        df0 = fft.FFTDF(cell1)
        odf = aft.AFTDF(cell1)
        nao = cell1.nao_nr()
        numpy.random.seed(5)
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri_mo0 = df0.get_mo_eri((mo,)*4, kpts)
        eri_mo1 = odf.get_mo_eri((mo,)*4, kpts)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        kpts_t = (kpts[2],kpts[3],kpts[0],kpts[1])
        eri_mo2 = df0.get_mo_eri((mo,)*4, kpts_t)
        eri_mo2 = eri_mo2.reshape((nao,)*4).transpose(2,3,0,1).reshape(nao**2,-1)
        self.assertTrue(np.allclose(eri_mo2, eri_mo0, atol=1e-7, rtol=1e-7))

        eri_mo0 = df0.get_mo_eri((mo,)*4, (kpts[0],)*4)
        eri_mo1 = odf.get_mo_eri((mo,)*4, (kpts[0],)*4)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        eri_mo0 = df0.get_mo_eri((mo,)*4, (kpts[0],kpts[1],kpts[1],kpts[0],))
        eri_mo1 = odf.get_mo_eri((mo,)*4, (kpts[0],kpts[1],kpts[1],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        eri_mo0 = df0.get_mo_eri((mo,)*4, (kpt0,kpt0,kpts[0],kpts[0],))
        eri_mo1 = odf.get_mo_eri((mo,)*4, (kpt0,kpt0,kpts[0],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        eri_mo0 = df0.get_mo_eri((mo,)*4, (kpts[0],kpts[0],kpt0,kpt0,))
        eri_mo1 = odf.get_mo_eri((mo,)*4, (kpts[0],kpts[0],kpt0,kpt0,))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        mo1 = mo[:,:nao//2+1]
        eri_mo0 = df0.get_mo_eri((mo1,mo,mo,mo1), (kpts[0],)*4)
        eri_mo1 = odf.get_mo_eri((mo1,mo,mo,mo1), (kpts[0],)*4)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

        eri_mo0 = df0.get_mo_eri((mo1,mo,mo1,mo), (kpts[0],kpts[1],kpts[1],kpts[0],))
        eri_mo1 = odf.get_mo_eri((mo1,mo,mo1,mo), (kpts[0],kpts[1],kpts[1],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-7, rtol=1e-7))

    def test_init_aft_1d(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.a = np.eye(3) * 2.5
        cell.dimension = 1
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.mesh = [3, 3, 3]
        cell.build()
        f = aft.AFTDF(cell)
        np.random.seed(1)
        f.kpts = np.random.random((4,3))
        f.check_sanity()

    def test_check_kpts(self):
        aft_obj = aft.AFTDF(cell)
        aft_obj.kpts = np.ones(3)
        ks, is_single_kpt = aft._check_kpts(aft_obj, None)
        assert ks.ndim == 2
        assert is_single_kpt
        ks, is_single_kpt = aft._check_kpts(aft_obj, np.ones((1, 3)))
        assert ks.ndim == 2
        assert not is_single_kpt
        ks, is_single_kpt = aft._check_kpts(aft_obj, np.ones(3))
        assert ks.ndim == 2
        assert is_single_kpt

        aft_obj.kpts = np.ones((1, 3))
        ks, is_single_kpt = aft._check_kpts(aft_obj, None)
        assert ks.ndim == 2
        assert not is_single_kpt
        ks, is_single_kpt = aft._check_kpts(aft_obj, np.ones((1, 3)))
        assert ks.ndim == 2
        assert not is_single_kpt
        ks, is_single_kpt = aft._check_kpts(aft_obj, np.ones(3))
        assert ks.ndim == 2
        assert is_single_kpt

if __name__ == '__main__':
    print("Full Tests for aft")
    unittest.main()
