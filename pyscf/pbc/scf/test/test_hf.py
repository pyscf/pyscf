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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf.scf import atom_hf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf
import pyscf.pbc.scf as pscf
from pyscf.pbc import df as pdf

def setUpModule():
    global cell, mf, kmf
    L = 4
    n = 21
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               verbose = 7,
               output = '/dev/null',
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]]})

    mf = pbchf.RHF(cell, exxdiv='ewald').run()
    kmf = pscf.KRHF(cell, [[0,0,0]], exxdiv='ewald').run()

def tearDownModule():
    global cell, mf, kmf
    cell.stdout.close()
    del cell, mf, kmf

class KnownValues(unittest.TestCase):
    def test_hcore(self):
        h1ref = pbchf.get_hcore(cell)
        h1 = pbchf.RHF(cell).get_hcore()
        self.assertAlmostEqual(abs(h1-h1ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(h1), 0.14116483012673137, 8)

        cell1 = cell.copy()
        cell1.ecp = {'He': (2, ((-1, (((7.2, .3),),)),))}
        cell1.build(0, 0)
        kpt = numpy.ones(3) * .5
        h1ref = pbchf.get_hcore(cell1, kpt)
        h1 = pbchf.RHF(cell1).get_hcore(kpt=kpt)
        self.assertAlmostEqual(abs(h1-h1ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(h1), -2.708431894877279-0.395390980665125j, 8)

        h1 = pscf.KRHF(cell1).get_hcore(kpts=[kpt])
        self.assertEqual(h1.ndim, 3)
        self.assertAlmostEqual(abs(h1[0]-h1ref).max(), 0, 9)

    def test_rhf_vcut_sph(self):
        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.1379172088570595, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_ewald(self):
        self.assertAlmostEqual(mf.e_tot, -4.3511582284698633, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)
        self.assertAlmostEqual(mf.e_tot, kmf.e_tot, 8)

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1, c1 = mf.get_bands(kpts_band)
        e0, c0 = kmf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1[0]), -6.2986775452228283, 7)
        self.assertAlmostEqual(lib.fp(e1[1]), -7.6616273746782362, 7)

    def test_rhf_exx_ewald_with_kpt(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        kmf = pscf.KRHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpt_band = numpy.random.random(3)
        e1, c1 = mf.get_bands(kpt_band)
        e0, c0 = kmf.get_bands(kpt_band)
        self.assertAlmostEqual(abs(e0-e1).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1), -6.8312867098806249, 7)

    def test_rhf_exx_None(self):
        mf = pbchf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_init_guess_by_chkfile(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        mf.chkfile = tempfile.NamedTemporaryFile().name
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.132445328608581, 7)

        mf1 = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        mf1.diis = None
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -4.291854736401251, 7)
        self.assertTrue(mf1.mo_coeff.dtype == numpy.double)

    def test_uhf_exx_ewald(self):
        mf = pscf.UHF(cell, exxdiv='ewald')
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582287379111, 7)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.double)

        kmf = pscf.KUHF(cell, [[0,0,0]], exxdiv='ewald')
        kmf.init_guess = 'hcore'
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1a, e1b = mf.get_bands(kpts_band)[0]
        e0a, e0b = kmf.get_bands(kpts_band)[0]
        self.assertAlmostEqual(abs(e0a[0]-e1a[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0a[1]-e1a[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[0]-e1b[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[1]-e1b[1]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(e1a[0]), -6.2986775452228283, 5)
        self.assertAlmostEqual(lib.fp(e1a[1]), -7.6616273746782362, 5)

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pscf.UHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 7)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.complex128)

        kmf = pscf.KUHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1a, e1b = mf.get_bands(kpts_band)[0]
        e0a, e0b = kmf.get_bands(kpts_band)[0]
        self.assertAlmostEqual(abs(e0a[0]-e1a[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0a[1]-e1a[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[0]-e1b[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[1]-e1b[1]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(e1a[0]), -6.8312867098806249, 5)
        self.assertAlmostEqual(lib.fp(e1a[1]), -6.1120214505413086, 5)

    def test_ghf_exx_ewald(self):
        mf = pscf.GHF(cell, exxdiv='ewald')
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582287379111, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        kmf = pscf.KGHF(cell, [[0,0,0]], exxdiv='ewald')
        kmf.init_guess = 'hcore'
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

#        # test bands
#        numpy.random.seed(1)
#        kpts_band = numpy.random.random((2,3))
#        e1, c1 = mf.get_bands(kpts_band)
#        e0, c0 = kmf.get_bands(kpts_band)
#        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
#        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
#        self.assertAlmostEqual(lib.fp(e1[0]), -6.2986775452228283, 7)
#        self.assertAlmostEqual(lib.fp(e1[1]), -7.6616273746782362, 7)

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pscf.GHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 7)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        kmf = pscf.KGHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

#        # test bands
#        numpy.random.seed(1)
#        kpts_band = numpy.random.random((2,3))
#        e1, c1 = mf.get_bands(kpts_band)
#        e0, c0 = kmf.get_bands(kpts_band)
#        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
#        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
#        self.assertAlmostEqual(lib.fp(e1[0]), -6.8312867098806249, 7)
#        self.assertAlmostEqual(lib.fp(e1[1]), -6.1120214505413086, 7)

    def test_rhf_0d(self):
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = numpy.eye(3)*5,
                   atom = '''He 2 2 2; He 2 2 3''',
                   dimension = 0,
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    [0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]]})
        eref = cell.to_mol().RHF().kernel()

        mf = cell.RHF()
        mf.with_df = pdf.AFTDF(cell)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, eref, 4)

        eref = cell.to_mol().RHF().density_fit().kernel()
        e1 = cell.RHF().density_fit().kernel()
        self.assertAlmostEqual(e1, eref, 9)

        cell = pbcgto.Cell()
        cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                      'C' :'gth-szv',}
        cell.pseudo = {'C':'gth-pade',
                       'He': pbcgto.pseudo.parse('''He
        2
         0.40000000    3    -1.98934751    -0.75604821    0.95604821
        2
         0.29482550    3     1.23870466    .855         .3
                                           .71         -1.1
                                                        .9
         0.32235865    2     2.25670239    -0.39677748
                                            0.93894690
                                                     ''')}
        cell.a = numpy.eye(3)
        cell.dimension = 0
        cell.build()
        mf = pscf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.run()

        mol = cell.to_mol()
        mf1 = mol.RHF().run()
        self.assertAlmostEqual(mf1.e_tot, -5.66198034773817, 8)
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 4)

    def test_rhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L*5,0],[0,0,L*5]],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.245417718, 6)

    def test_rhf_2d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L,0],[0,0,L*5]],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.2683850732448168, 5)

    def test_rhf_2d_fft(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L,0],[0,0,10]],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell, exxdiv='ewald')
        mf.with_df = pdf.FFTDF(cell)
        mf.with_df.mesh = cell.mesh
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.268385073966333, 7)

        mf1 = pbchf.RHF(cell, exxdiv='ewald')
        mf1.with_df = pdf.FFTDF(cell)
        mf1.with_df.mesh = cell.mesh
        mf1.direct_scf = True
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -3.268385073966333, 7)

        mf2 = pbchf.RHF(cell, exxdiv=None)
        mf2.with_df = pdf.FFTDF(cell)
        mf2.with_df.mesh = cell.mesh
        mf2.direct_scf = True
        e2 = mf2.kernel()
        self.assertAlmostEqual(e2, -1.3182526139263366, 7)

    def test_uhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = numpy.eye(3)*4,
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pscf.UHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.245417718, 6)

    def test_ghf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = numpy.eye(3)*4,
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   rcut = 7.427535697575829,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pscf.GHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.245417718, 6)

    def test_get_veff(self):
        mf = pscf.RHF(cell)
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao)) + numpy.random.random((nao,nao))*1j
        dm = dm + dm.conj().T
        v11 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([.25,.25,.25]))
        v12 = mf.get_veff(cell, dm, kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v13 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v14 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.make_kpts([2,1,1]))
        self.assertTrue(v11.dtype == numpy.complex128)
        self.assertTrue(v12.dtype == numpy.complex128)

        mf = pscf.UHF(cell)
        v21 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([.25,.25,.25]))
        dm = [dm*.5,dm*.5]
        v22 = mf.get_veff(cell, dm, kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v23 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v24 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.make_kpts([2,1,1]))
        self.assertAlmostEqual(abs(v11-v21).max(), 0, 9)
        self.assertAlmostEqual(abs(v12-v22).max(), 0, 9)
        self.assertAlmostEqual(abs(v13-v23).max(), 0, 9)
        self.assertAlmostEqual(abs(v14-v24).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(v11), -0.30110964334164825+0.81409418199767414j, 8)
        self.assertAlmostEqual(lib.fp(v12), -2.1601376488983997-9.4070613374115908j, 8)

    def test_init(self):
        from pyscf.pbc import dft
        cell_u = cell.copy()
        cell_u.spin = 2
        self.assertTrue(isinstance(pscf.RKS  (cell  ), dft.rks.RKS    ))
        self.assertTrue(isinstance(pscf.RKS  (cell_u), dft.roks.ROKS  ))
        self.assertTrue(isinstance(pscf.UKS  (cell  ), dft.uks.UKS    ))
        self.assertTrue(isinstance(pscf.ROKS (cell  ), dft.roks.ROKS  ))
        self.assertTrue(isinstance(pscf.KS   (cell  ), dft.rks.RKS    ))
        self.assertTrue(isinstance(pscf.KS   (cell_u), dft.uks.UKS    ))
        self.assertTrue(isinstance(pscf.KRKS (cell  ), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KRKS (cell_u), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KUKS (cell  ), dft.kuks.KUKS  ))
        self.assertTrue(isinstance(pscf.KROKS(cell  ), dft.kroks.KROKS))
        self.assertTrue(isinstance(pscf.KKS  (cell  ), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KKS  (cell_u), dft.kuks.KUKS  ))

        self.assertTrue(isinstance(pscf.RHF  (cell  ), pscf.hf.RHF     ))
        self.assertTrue(isinstance(pscf.RHF  (cell_u), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.KRHF (cell  ), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.KRHF (cell_u), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.UHF  (cell  ), pscf.uhf.UHF    ))
        self.assertTrue(isinstance(pscf.KUHF (cell_u), pscf.kuhf.KUHF  ))
        self.assertTrue(isinstance(pscf.GHF  (cell  ), pscf.ghf.GHF    ))
        self.assertTrue(isinstance(pscf.KGHF (cell_u), pscf.kghf.KGHF  ))
        self.assertTrue(isinstance(pscf.ROHF (cell  ), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.ROHF (cell_u), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.KROHF(cell  ), pscf.krohf.KROHF))
        self.assertTrue(isinstance(pscf.KROHF(cell_u), pscf.krohf.KROHF))
        self.assertTrue(isinstance(pscf.HF   (cell  ), pscf.hf.RHF     ))
        self.assertTrue(isinstance(pscf.HF   (cell_u), pscf.uhf.UHF    ))
        self.assertTrue(isinstance(pscf.KHF  (cell  ), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.KHF  (cell_u), pscf.kuhf.KUHF  ))

    def test_makov_payne_correction(self):
        from pyscf.pbc.dft import gen_grid
        de = pbchf.makov_payne_correction(mf)
        self.assertAlmostEqual(de[0], -0.1490687416177664, 2)
        self.assertAlmostEqual(de[0], de[1], 7)
        self.assertAlmostEqual(de[0], de[2], 7)

        dm = mf.make_rdm1()
        grids = gen_grid.UniformGrids(cell)
        rho = pscf.hf.get_rho(mf, dm, grids)
        log = lib.logger.new_logger(mf)
        center = pscf.hf._search_dipole_gauge_origin(cell, grids, rho, log)
        self.assertAlmostEqual(abs(center - [1.75, 2, 2]).max(), 0, 2)

        dip = mf.dip_moment(cell, dm)
        self.assertAlmostEqual(abs(dip).max(), 0, 1)

    def test_init_guess_by_1e(self):
        dm = mf.get_init_guess(key='1e')
        self.assertAlmostEqual(lib.fp(dm), 0.025922864381755062, 6)

        dm = kmf.get_init_guess(key='1e')
        self.assertEqual(dm.ndim, 3)
        self.assertAlmostEqual(lib.fp(dm), 0.025922864381755062, 6)

    def test_init_guess_by_minao(self):
        with lib.temporary_env(cell, dimension=1):
            dm = mf.get_init_guess(key='minao')
            kdm = kmf.get_init_guess(key='minao')

        self.assertAlmostEqual(lib.fp(dm), -1.714952331211208, 8)

        self.assertEqual(kdm.ndim, 3)
        self.assertAlmostEqual(lib.fp(kdm), -1.714952331211208, 8)

    def test_init_guess_by_atom(self):
        with lib.temporary_env(cell, dimension=1):
            dm = mf.get_init_guess(key='atom')
            kdm = kmf.get_init_guess(key='atom')

        self.assertAlmostEqual(lib.fp(dm), 0.18074522075843902, 7)

        self.assertEqual(kdm.ndim, 3)
        self.assertAlmostEqual(lib.fp(dm), 0.18074522075843902, 7)

    def test_atom_hf_with_pp(self):
        mol = pbcgto.Cell()
        mol.build(
            verbose = 7,
            output = '/dev/null',
            atom  = 'O 0 0 0; H 0 0 -1; H 0 0 1',
            a = [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            basis = 'gth-dzvp',
            pseudo = 'gth-pade')
        scf_result = atom_hf.get_atm_nrhf(mol)
        self.assertAlmostEqual(scf_result['O'][0], -15.193243796069835, 9)
        self.assertAlmostEqual(scf_result['H'][0], -0.49777509423571864, 9)

    def test_jk(self):
        nao = cell.nao
        numpy.random.seed(2)
        dm = numpy.random.random((2,nao,nao)) + .5j*numpy.random.random((2,nao,nao))
        dm = dm + dm.conj().transpose(0,2,1)
        ref = pbchf.get_jk(mf, cell, dm)
        vj, vk = mf.get_jk_incore(cell, dm)
        self.assertAlmostEqual(abs(vj - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vk - ref[1]).max(), 0, 9)

    def test_analyze(self):
        rpop, rchg = mf.analyze()[0]
        self.assertAlmostEqual(lib.fp(rpop), 0.0110475, 4)
        self.assertAlmostEqual(abs(rchg).max(), 0, 7)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
