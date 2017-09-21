#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf
import pyscf.pbc.scf as pscf
from pyscf.pbc import df as pdf

L = 4
n = 10
cell = pbcgto.Cell()
cell.build(unit = 'B',
           verbose = 7,
           output = '/dev/null',
           a = ((L,0,0),(0,L,0),(0,0,L)),
           gs = [n,n,n],
           atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                   ['He', (L/2.   ,L/2.,L/2.+.5)]],
           basis = { 'He': [[0, (0.8, 1.0)],
                            [0, (1.0, 1.0)],
                            [0, (1.2, 1.0)]]})

class KnowValues(unittest.TestCase):
    def test_rhf_vcut_sph(self):
        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.1379172088570595, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_ewald(self):
        mf = pbchf.RHF(cell, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582284698633, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='ewald')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='ewald')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_None(self):
        mf = pbchf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_init_guess_by_chkfile(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        dm1 = mf.make_rdm1()

        mf1 = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 8)
        self.assertTrue(mf1.mo_coeff.dtype == numpy.double)

    def test_uhf_exx_ewald(self):
        mf = pscf.UHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 8)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.double)

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 8)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.complex128)

        mf = pscf.UHF(cell, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582287379111, 8)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.double)

#    def test_rhf_0d(self):
#        from pyscf.df import mdf_jk
#        from pyscf.scf import hf
#        L = 4
#        cell = pbcgto.Cell()
#        cell.build(unit = 'B',
#                   a = numpy.eye(3)*L*5,
#                   gs = [10]*3,
#                   atom = '''He 2 2 2; He 2 2 3''',
#                   dimension = 0,
#                   verbose = 0,
#                   basis = { 'He': [[0, (0.8, 1.0)],
#                                    [0, (1.0, 1.0)],
#                                    [0, (1.2, 1.0)]]})
#        mol = cell.to_mol()
#        mf = mdf_jk.density_fit(hf.RHF(mol))
#        mf.with_df.gs = [10]*3
#        mf.with_df.auxbasis = {'He':[[0, (1e6, 1)]]}
#        mf.with_df.charge_constraint = False
#        mf.with_df.metric = 'S'
#        eref = mf.kernel()
#
#        mf = pbchf.RHF(cell)
#        mf.with_df = pdf.AFTDF(cell)
#        mf.exxdiv = None
#        mf.get_hcore = lambda *args: hf.get_hcore(mol)
#        mf.energy_nuc = lambda *args: mol.energy_nuc()
#        e1 = mf.kernel()
#        self.assertAlmostEqual(e1, eref, 8)

    def test_rhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L*5,0],[0,0,L*5]],
                   gs = [5,10,10],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.gs = cell.gs
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.24497234871167, 5)

    def test_rhf_2d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L,0],[0,0,L*5]],
                   gs = [5,5,10],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.gs = cell.gs
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.2681555164454039, 5)

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
        self.assertTrue(v11.dtype == numpy.complex128)
        self.assertTrue(v12.dtype == numpy.complex128)

        mf = pscf.UHF(cell)
        v21 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([.25,.25,.25]))
        dm = [dm*.5,dm*.5]
        v22 = mf.get_veff(cell, dm, kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v23 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        self.assertAlmostEqual(abs(v11-v21).max(), 0, 9)
        self.assertAlmostEqual(abs(v12-v22).max(), 0, 9)
        self.assertAlmostEqual(abs(v13-v23).max(), 0, 9)
        self.assertAlmostEqual(lib.finger(v11), -0.30110964334164825+0.81409418199767414j, 9)
        self.assertAlmostEqual(lib.finger(v12), -2.1601376488983997-9.4070613374115908j, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
