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
import ctypes
import numpy
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.gto import ecp
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.gto import ewald_methods


def setUpModule():
    global cl, cl1, L, n
    L = 1.5
    n = 41
    cl = pgto.Cell()
    cl.build(
        a = [[L,0,0], [0,L,0], [0,0,L]],
        mesh = [n,n,n],
        atom = 'He %f %f %f' % ((L/2.,)*3),
        basis = 'ccpvdz')

    numpy.random.seed(1)
    cl1 = pgto.Cell()
    cl1.build(a = numpy.random.random((3,3)).T,
              precision = 1e-9,
              mesh = [n,n,n],
              atom ='''He .1 .0 .0
                       He .5 .1 .0
                       He .0 .5 .0
                       He .1 .3 .2''',
              basis = 'ccpvdz')

def tearDownModule():
    global cl, cl1
    del cl, cl1

class KnownValues(unittest.TestCase):
    def test_nimgs(self):
        self.assertTrue(list(cl.get_nimgs(9e-1)), [1,1,1])
        self.assertTrue(list(cl.get_nimgs(1e-2)), [2,2,2])
        self.assertTrue(list(cl.get_nimgs(1e-4)), [3,3,3])
        self.assertTrue(list(cl.get_nimgs(1e-6)), [4,4,4])
        self.assertTrue(list(cl.get_nimgs(1e-9)), [5,5,5])

    def test_Gv(self):
        a = cl1.get_Gv()
        self.assertAlmostEqual(lib.fp(a), -99.791927068519939, 10)

    def test_SI(self):
        a = cl1.get_SI()
        self.assertAlmostEqual(lib.fp(a), (16.506917823339265+1.6393578329869585j), 10)

        np.random.seed(2)
        Gv = np.random.random((5,3))
        a = cl1.get_SI(Gv)
        self.assertAlmostEqual(lib.fp(a), (0.65237631847195221-1.5736011413431059j), 10)

    def test_mixed_basis(self):
        cl = pgto.Cell()
        cl.build(
            a = [[L,0,0], [0,L,0], [0,0,L]],
            mesh = [n,n,n],
            atom = 'C1 %f %f %f; C2 %f %f %f' % ((L/2.,)*6),
            basis = {'C1':'ccpvdz', 'C2':'gthdzv'})

    def test_dumps_loads(self):
        cl1.loads(cl1.dumps())
        # see issue 2026
        from pyscf.pbc.tools.pbc import super_cell
        sc = super_cell(cl1, [1,1,1])
        sc.dumps()

    def test_get_lattice_Ls(self):
        #self.assertEqual(cl1.get_lattice_Ls([0,0,0]).shape, (1  , 3))
        #self.assertEqual(cl1.get_lattice_Ls([1,1,1]).shape, (13 , 3))
        #self.assertEqual(cl1.get_lattice_Ls([2,2,2]).shape, (57 , 3))
        #self.assertEqual(cl1.get_lattice_Ls([3,3,3]).shape, (137, 3))
        #self.assertEqual(cl1.get_lattice_Ls([4,4,4]).shape, (281, 3))
        #self.assertEqual(cl1.get_lattice_Ls([5,5,5]).shape, (493, 3))

        cell = pgto.M(atom = '''
        C 0.000000000000  0.000000000000  0.000000000000
        C 1.685068664391  1.685068664391  1.685068664391''',
        unit='B',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        a = '''
        0.000000000  3.370137329  3.370137329
        3.370137329  0.000000000  3.370137329
        3.370137329  3.370137329  0.000000000''',
        mesh = [15]*3)
        rcut = max([cell.bas_rcut(ib, 1e-8) for ib in range(cell.nbas)])
        self.assertEqual(cell.get_lattice_Ls(rcut=rcut).shape, (1439, 3))
        #rcut = max([cell.bas_rcut(ib, 1e-9) for ib in range(cell.nbas)])
        #self.assertEqual(cell.get_lattice_Ls(rcut=rcut).shape, (1499, 3))

    def test_fractional_coordinates(self):
        cell = pgto.M(atom = '''
        C 0 0 0
        C .25 .25 .25''',
        unit='B', basis = 'gth-dzvp', pseudo = 'gth-pade',
        fractional=True,
        a = '''
        0.000000000  3.370137329  3.370137329
        3.370137329  0.000000000  3.370137329
        3.370137329  3.370137329  0.000000000''')
        #[[0.         0.         0.        ]
        #  [1.68506866 1.68506866 1.68506866]]
        self.assertAlmostEqual(lib.fp(cell.atom_coords()), -2.2916494573514545, 14)

    def test_ewald(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        Lx = Ly = Lz = 5.
        cell.a = numpy.diag([Lx,Ly,Lz])
        cell.mesh = numpy.array([41]*3)
        cell.atom = [['He', (2, 0.5*Ly, 0.5*Lz)],
                     ['He', (3, 0.5*Ly, 0.5*Lz)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        self.assertAlmostEqual(cell.ewald(0.2, 30), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(1  , 30), -0.468640671931, 9)

        cell = pgto.Cell()
        numpy.random.seed(10)
        cell.a = numpy.random.random((3,3))*2 + numpy.eye(3) * 2
        cell.mesh = [41]*3
        cell.atom = [['He', (1, 1, 2)],
                     ['He', (3, 2, 1)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        self.assertAlmostEqual(cell.ewald(1, 20), -2.3711356723457615, 9)
        self.assertAlmostEqual(cell.ewald(2, 10), -2.3711356723457615, 9)
        self.assertAlmostEqual(cell.ewald(2,  5), -2.3711356723457615, 9)

    def test_ewald_vs_supercell(self):
        a  = 4.1705
        cell = pgto.Cell()
        cell.a = a * np.asarray([
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1.0]])

        cell.atom = [
                ['Ni', [0, 0, 0]],
                ['Ni', [a, a, a]]]
        cell.precision = 1e-8
        cell.build()
        e_nuc_1 = cell.energy_nuc()
        self.assertAlmostEqual(e_nuc_1, -456.0950359594, 8)

        celldims = [2, 1, 1]
        scell = pbctools.super_cell(cell, celldims)
        e_nuc_2 = scell.energy_nuc() / np.prod(celldims)
        self.assertAlmostEqual(e_nuc_1, e_nuc_2, 8)

        celldims = [2, 2, 1]
        scell = pbctools.super_cell(cell, celldims)
        e_nuc_2 = scell.energy_nuc() / np.prod(celldims)
        self.assertAlmostEqual(e_nuc_1, e_nuc_2, 8)

    def test_ewald_2d_inf_vacuum(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [9,9,60]
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.dimension = 2
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.rcut = 3.6
        cell.build()
        # FIXME: why python 3.8 generates different value at 4th decimal place
        self.assertAlmostEqual(cell.ewald(), 3898143.7149599474, 2)

    def test_ewald_1d_inf_vacuum(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [9,60,60]
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.dimension = 1
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.rcut = 3.6
        cell.build()
        self.assertAlmostEqual(cell.ewald(), 70.875156940393225, 4)

    def test_ewald_0d_inf_vacuum(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3)
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [60] * 3
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.dimension = 0
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.build()
        eref = cell.to_mol().energy_nuc()
        self.assertAlmostEqual(cell.ewald(), eref, 2)

    def test_ewald_2d(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [9,9,60]
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.dimension = 2
        cell.rcut = 3.6
        cell.build()
        self.assertAlmostEqual(cell.ewald(), -5.1194779101355596, 9)

        a = numpy.eye(3) * 3
        a[0,1] = .2
        c = pgto.M(atom='H 0 0.1 0; H 1.1 2.0 0; He 1.2 .3 0.2',
                   a=a, dimension=2, verbose=0)
        self.assertAlmostEqual(c.ewald(), -3.0902098018260418, 9)

#    def test_ewald_1d(self):
#        cell = pgto.Cell()
#        cell.a = numpy.eye(3) * 4
#        cell.atom = 'He 0 0 0; He 0 1 1'
#        cell.unit = 'B'
#        cell.mesh = [9,60,60]
#        cell.verbose = 0
#        cell.dimension = 1
#        cell.rcut = 3.6
#        cell.build()
#        self.assertAlmostEqual(cell.ewald(), 70.875156940393225, 8)
#
#    def test_ewald_0d(self):
#        cell = pgto.Cell()
#        cell.a = numpy.eye(3)
#        cell.atom = 'He 0 0 0; He 0 1 1'
#        cell.unit = 'B'
#        cell.mesh = [60] * 3
#        cell.verbose = 0
#        cell.dimension = 0
#        cell.build()
#        eref = cell.to_mol().energy_nuc()
#        self.assertAlmostEqual(cell.ewald(), eref, 2)

    def test_particle_mesh_ewald(self):
        cell = pgto.Cell()
        cell.a = np.diag([10.,]*3)
        cell.atom = '''
            O          5.84560        5.21649        5.10372
            H          6.30941        5.30070        5.92953
            H          4.91429        5.26674        5.28886
        '''
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.build()

        cell1 = cell.copy()
        cell1.use_particle_mesh_ewald = True
        cell1.build()

        e0 = cell.ewald()
        e1 = cell1.ewald()
        self.assertAlmostEqual(e0, e1, 6)

        g0 = ewald_methods.ewald_nuc_grad(cell)
        g1 = ewald_methods.ewald_nuc_grad(cell1)
        self.assertAlmostEqual(abs(g1-g0).max(), 0, 6)

    def test_pbc_intor(self):
        numpy.random.seed(12)
        kpts = numpy.random.random((4,3))
        kpts[0] = 0
        #self.assertEqual(list(cl1.nimgs), [34,23,21])
        s0 = cl1.pbc_intor('int1e_ovlp_sph', hermi=0, kpts=kpts)
        self.assertAlmostEqual(lib.fp(s0[0]), 492.30658304804126, 4)
        self.assertAlmostEqual(lib.fp(s0[1]), 37.812956255000756-28.972806230140314j, 4)
        self.assertAlmostEqual(lib.fp(s0[2]),-26.113285893260819-34.448501789693566j, 4)
        self.assertAlmostEqual(lib.fp(s0[3]), 186.58921213429491+123.90133823378201j, 4)

        s1 = cl1.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts[0])
        self.assertAlmostEqual(lib.fp(s1), 492.30658304804126, 4)

    def test_ecp_pseudo(self):
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Cu 0 0 1; Na 0 1 0',
            ecp = {'Na':'lanl2dz'},
            pseudo = {'Cu': 'gthbp'})
        self.assertTrue(all(cell._ecpbas[:,0] == 1))

    def test_ecp_int(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 8
        cell.mesh = [11] * 3
        cell.atom='''Na 0. 0. 0.
                     H  0.  0.  1.'''
        cell.basis={'Na':'lanl2dz', 'H':'sto3g'}
        cell.ecp = {'Na':'lanl2dz'}
        cell.build()
        v1 = ecp.ecp_int(cell)
        mol = cell.to_mol()
        v0 = mol.intor('ECPscalar_sph')
        self.assertAlmostEqual(abs(v0 - v1).sum(), 0.029005926114411891, 8)
        self.assertAlmostEqual(lib.fp(v1), -0.20831852433927503, 8)

        cell = pgto.M(
            verbose = 0,
            a = np.eye(3)*6,
            atom = 'Na 1 0 1; Cl 5 4 4',
            ecp = 'lanl2dz',
            basis = [[0, [1, 1]]])
        v1 = ecp.ecp_int(cell)
        mol = cell.to_mol()
        v0 = mol.intor('ECPscalar_sph')
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(v1), -1.225444628445373, 8)

        cell = pgto.M(a = '''0     2.445 2.445
                     2.445 0     2.445
                     2.445 2.445 0 ''',
                     atom = 'U 0.0 0.0 0.0',
                     basis = [[0, [.3, 1]], [2, [.2, 1]]],
                     ecp = {'U': '''U nelec 60
                            U S
                            2   16.414038690   536.516627780
                            U P
                            2   9.060556060   169.544924650
                            '''},
                     precision = 1e-7,
        )
        nk = [4] * 3
        kpts = cell.make_kpts(nk)
        h1 = ecp.ecp_int(cell, kpts)
        self.assertAlmostEqual(lib.fp(h1), 4.160881841456467, 7)

    def test_ecp_keyword_in_pseudo(self):
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = 'lanl2dz',
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, 'lanl2dz')
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = {'na': 'lanl2dz'},
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'na': 'lanl2dz', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = {'S': 'gthbp', 'na': 'lanl2dz'},
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'na': 'lanl2dz', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'S': 'gthbp', 'O': 'gthbp'})

    def test_pseudo_suffix(self):
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Mg 0 0 1',
            pseudo = {'Mg': 'gth-lda'})
        self.assertEqual(cell.atom_nelec_core(0), 2)

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Mg 0 0 1',
            pseudo = {'Mg': 'gth-lda q2'})
        self.assertEqual(cell.atom_nelec_core(0), 10)

    def pbc_intor_symmetry(self):
        a = cl1.lattice_vectors()
        b = numpy.linalg.inv(a).T * (numpy.pi*2)
        kpts = numpy.random.random((4,3))
        kpts[1] = b[0]+b[1]+b[2]-kpts[0]
        kpts[2] = b[0]-b[1]-b[2]-kpts[0]
        kpts[3] = b[0]-b[1]+b[2]+kpts[0]
        s = cl1.pbc_intor('int1e_ovlp', kpts=kpts)
        self.assertAlmostEqual(abs(s[0]-s[1].conj()).max(), 0, 12)
        self.assertAlmostEqual(abs(s[0]-s[2].conj()).max(), 0, 12)
        self.assertAlmostEqual(abs(s[0]-s[3]       ).max(), 0, 12)

    def test_basis_truncation(self):
        b = pgto.basis.load('gthtzvp@3s1p', 'C')
        self.assertEqual(len(b), 2)
        self.assertEqual(len(b[0][1]), 4)
        self.assertEqual(len(b[1][1]), 2)

    def test_getattr(self):
        from pyscf.pbc import scf, dft, cc, tdscf
        cell = pgto.M(atom='He', a=np.eye(3)*4, basis={'He': [[0, (1, 1)]]})
        self.assertEqual(cell.HF().__class__, scf.HF(cell).__class__)
        self.assertEqual(cell.KS().__class__, dft.KS(cell).__class__)
        self.assertEqual(cell.UKS().__class__, dft.UKS(cell).__class__)
        self.assertEqual(cell.KROHF().__class__, scf.KROHF(cell).__class__)
        self.assertEqual(cell.KKS().__class__, dft.KKS(cell).__class__)
        self.assertEqual(cell.CCSD().__class__, cc.ccsd.RCCSD)
        self.assertEqual(cell.TDA().__class__, tdscf.rhf.TDA)
        self.assertEqual(cell.TDBP86().__class__, tdscf.rks.CasidaTDDFT)
        self.assertEqual(cell.TDB3LYP().__class__, tdscf.rks.TDDFT)
        self.assertEqual(cell.KCCSD().__class__, cc.kccsd_rhf.KRCCSD)
        self.assertEqual(cell.KTDA().__class__, tdscf.krhf.TDA)
        self.assertEqual(cell.KTDBP86().__class__, tdscf.krks.TDDFT)
        self.assertRaises(AttributeError, lambda: cell.xyz)
        self.assertRaises(AttributeError, lambda: cell.TDxyz)

        cell = pgto.M(atom='He', charge=1, spin=1, a=np.eye(3)*4, basis={'He': [[0, (1, 1)]]})
        self.assertTrue(cell.HF().__class__, scf.uhf.UHF)
        self.assertTrue(cell.KS().__class__, dft.uks.UKS)
        self.assertTrue(cell.KKS().__class__, dft.kuks.KUKS)
        self.assertTrue(cell.CCSD().__class__, cc.ccsd.UCCSD)
        self.assertTrue(cell.TDA().__class__, tdscf.uhf.TDA)
        self.assertTrue(cell.TDBP86().__class__, tdscf.uks.CasidaTDDFT)
        self.assertTrue(cell.TDB3LYP().__class__, tdscf.uks.TDDFT)
        self.assertTrue(cell.KCCSD().__class__, cc.kccsd_uhf.KUCCSD)
        self.assertTrue(cell.KTDA().__class__, tdscf.kuhf.TDA)
        self.assertTrue(cell.KTDBP86().__class__, tdscf.kuks.TDDFT)

    def test_ghost(self):
        cell = pgto.Cell(
            atom = 'C 0 0 0; ghost 0 0 2',
            basis = {'C': 'sto3g', 'ghost': gto.basis.load('sto3g', 'H')},
            a = np.eye(3) * 3,
            pseudo = 'gth-pade',
        ).run()
        self.assertEqual(cell.nao_nr(), 6)

        cell = pgto.M(atom='''
        ghost-O     0.000000000     0.000000000     2.500000000
        X_H        -0.663641000    -0.383071000     3.095377000
        ghost.H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''',
        a=np.eye(3) * 3,
        pseudo={'default': 'gth-pade', 'ghost-O': 'gth-pade'},
        basis='gth-dzv')
        self.assertEqual(cell.nao_nr(), 24)  # 8 + 2 + 2 + 8 + 2 + 2
        self.assertTrue(len(cell._pseudo) == 3) # O, H, ghost-O in ecp

        cell = pgto.M(atom='''
        ghost-O     0.000000000     0.000000000     2.500000000
        X_H        -0.663641000    -0.383071000     3.095377000
        ghost.H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        ''',
        a=np.eye(3) * 3,
        pseudo='gth-pade',
        basis={'H': 'gth-dzv', 'o': 'gth-dzvp', 'ghost-O': 'gth-szv'})
        self.assertEqual(cell.nao_nr(), 21) # 4 + 2 + 2 + 13
        self.assertTrue(len(cell._pseudo) == 1)  # only O in ecp

    def test_exp_to_discard(self):
        cell = pgto.Cell(
            atom = 'Li 0 0 0; Li 1.5 1.5 1.5',
            a = np.eye(3) * 3,
            basis = "gth-dzvp",
            exp_to_discard = .1
        )
        cell.build()
        cell1 =  pgto.Cell(
            atom = 'Li@1 0 0 0; Li@2 1.5 1.5 1.5',
            a = np.eye(3) * 3,
            basis = "gth-dzvp",
            exp_to_discard = .1
        )
        cell1.build()
        for ib in range(len(cell._bas)):
            nprim = cell.bas_nprim(ib)
            nc = cell.bas_nctr(ib)
            es = cell.bas_exp(ib)
            es1 = cell1.bas_exp(ib)
            ptr = cell._bas[ib, gto.mole.PTR_COEFF]
            ptr1 = cell1._bas[ib, gto.mole.PTR_COEFF]
            cs = cell._env[ptr:ptr+nprim*nc]
            cs1 = cell1._env[ptr1:ptr1+nprim*nc]
            self.assertAlmostEqual(abs(es - es1).max(), 0, 15)
            self.assertAlmostEqual(abs(cs - cs1).max(), 0, 15)

    def test_conc_cell(self):
        cl1 = pgto.M(a=np.eye(3)*5, atom='Cu', basis='lanl2dz', ecp='lanl2dz', spin=None)
        cl2 = pgto.M(a=np.eye(3)*5, atom='Cs', basis='lanl2dz', ecp='lanl2dz', spin=None)
        cl3 = cl1 + cl2
        self.assertTrue(len(cl3._ecpbas), 20)
        self.assertTrue(len(cl3._bas), 12)
        self.assertTrue(len(cl3._atm), 8)
        self.assertAlmostEqual(abs(cl3.lattice_vectors() - cl1.lattice_vectors()).max(), 0, 12)

    def test_eval_gto(self):
        cell = pgto.M(a=np.eye(3)*4, atom='He 1 1 1', basis=[[2,(1,.5),(.5,.5)]], precision=1e-10)
        coords = cell.get_uniform_grids([10]*3, wrap_around=False)
        ao_value = cell.pbc_eval_gto("GTOval_sph", coords, kpts=cell.make_kpts([3]*3))
        self.assertAlmostEqual(lib.fp(ao_value), (-0.27594803231989179+0.0064644591759109114j), 9)

        cell = pgto.M(a=np.eye(3)*4, atom='He 1 1 1', basis=[[2,(1,.5),(.5,.5)]], precision=1e-10)
        coords = cell.get_uniform_grids([10]*3, wrap_around=False)
        ao_value = cell.pbc_eval_gto("GTOval_ip_cart", coords, kpts=cell.make_kpts([3]*3))
        self.assertAlmostEqual(lib.fp(ao_value), (0.38051517609460028+0.062526488684770759j), 9)

    def test_empty_cell(self):
        cell = pgto.M(a=np.eye(3)*4)
        Ls = pbctools.get_lattice_Ls(cell)
        self.assertEqual(abs(Ls-np.zeros([1,3])).max(), 0)

    def test_fromstring(self):
        ref = cl.atom_coords().copy()
        cell = pgto.Cell()
        cell.fromstring(cl.tostring('poscar'), 'vasp')
        r0 = cell.atom_coords()
        self.assertAlmostEqual(abs(ref - r0).max(), 0, 12)
        cell.fromstring(cl.tostring('xyz'), 'xyz')
        r0 = cell.atom_coords()
        self.assertAlmostEqual(abs(ref - r0).max(), 0, 12)

    def test_fromfile(self):
        ref = cl.atom_coords().copy()
        with tempfile.NamedTemporaryFile() as f:
            cl.tofile(f.name, 'xyz')
            cell = pgto.Cell()
            cell.fromfile(f.name, 'xyz')
            r1 = cell.atom_coords()
            self.assertAlmostEqual(abs(ref - r1).max(), 0, 12)

if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()
