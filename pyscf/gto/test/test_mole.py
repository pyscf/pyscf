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

import unittest
import tempfile
from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
import pyscf.lib.parameters as param
from pyscf.lib.exceptions import BasisNotFoundError, PointGroupSymmetryError

def setUpModule():
    global mol0, ftmp
    mol0 = gto.Mole()
    mol0.atom = [
        [1  , (0.,1.,1.)],
        ["O1", (0.,0.,0.)],
        [1  , (1.,1.,0.)], ]
    mol0.nucmod = { "O":'gaussian', 3:'g' }
    mol0.unit = 'ang'
    mol0.basis = {
        "O": [(0, 0, (15, 1)), ] + gto.etbs(((0, 4, 1, 1.8),
                                             (1, 3, 2, 1.8),
                                             (2, 2, 1, 1.8),)),
        "H": [(0, 0, (1, 1, 0), (3, 3, 1), (5, 1, 0)),
              (1, -2, (1, 1)), ]}
    mol0.symmetry = 1
    mol0.charge = 1
    mol0.spin = 1
    mol0.verbose = 7
    mol0.ecp = {'O1': 'lanl2dz'}
    ftmp = tempfile.NamedTemporaryFile()
    mol0.output = ftmp.name
    mol0.build()

def tearDownModule():
    global mol0, ftmp
    mol0.stdout.close()
    del mol0, ftmp

class KnownValues(unittest.TestCase):
    def test_intor_cross(self):
        mol1 = mol0.unpack(mol0.pack())
        mol1.symmetry = True
        mol1.unit = 'Ang'
        mol1.atom = '''
                1    0  1  .5*2
                O    0  0  0*np.exp(0)
                h    1  1  0'''
        mol1.basis = {'O': gto.basis.parse('''
C    S
   3047.5249000              0.0018347*1.0
    457.3695100              0.0140373*1.0
    103.9486900              0.0688426*1.0
     29.2101550              0.2321844*1.0
      9.2866630              0.4679413*1.0
      3.1639270              0.3623120*1.0
#     1.                     0.1
C    SP
      7.8682724             -0.1193324*1.0          0.0689991        
      1.8812885             -0.1608542*1.0          0.3164240        
      0.5442493              1.1434564*1.0          0.7443083        
C    SP
      0.1687144              1.0000000              1.0000000'''),
                      'H': '6-31g'}
        mol1.build()
        v = gto.mole.intor_cross('cint1e_ovlp_sph', mol0, mol1)
        self.assertAlmostEqual(numpy.linalg.norm(v), 3.6489423434168562, 1)

    def test_num_basis(self):
        self.assertEqual(mol0.nao_nr(), 34)
        self.assertEqual(mol0.nao_2c(), 64)

    def test_time_reversal_map(self):
        tao = [ -2,  1, -4,  3,  8, -7,  6, -5,-10,  9,-12, 11,-14, 13,-16, 15,-18, 17,
                20,-19, 24,-23, 22,-21, 26,-25, 30,-29, 28,-27, 32,-31, 36,-35, 34,-33,
               -40, 39,-38, 37,-46, 45,-44, 43,-42, 41,-50, 49,-48, 47,-56, 55,-54, 53,
               -52, 51,-58, 57,-60, 59, 64,-63, 62,-61]
        self.assertEqual(list(mol0.time_reversal_map()), tao)

    def test_check_sanity(self):
        mol1 = mol0.copy()
        mol1.x = None
        mol1.copy = None
        mol1.check_sanity()

    def test_nao_range(self):
        self.assertEqual(mol0.nao_nr_range(1,4), (2, 7))
        self.assertEqual(mol0.nao_2c_range(1,4), (4, 12))
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_nr()), 2151)
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_2c()), 4066)

    def test_search_bas(self):
        self.assertEqual(mol0.search_shell_id(1, 1), 7)
        self.assertRaises(RuntimeError, mol0.search_ao_nr, 1, 1, -1, 5)
        self.assertEqual(mol0.search_ao_nr(1, 1, -1, 4), 16)
        mol0.cart = True
        self.assertEqual(mol0.search_ao_nr(2, 1, -1, 1), 30)
        mol0.cart = False

    def test_atom_types(self):
        atoms = [['H0', ( 0, 0, 0)],
                 ['H1', ( 0, 0, 0)],
                 ['H',  ( 0, 0, 0)],
                 ['H3', ( 0, 0, 0)]]
        basis = {'H':'sto3g', 'H1': '6-31g'}
        atmgroup = gto.mole.atom_types(atoms, basis)
        self.assertEqual(atmgroup, {'H': [0, 2, 3], 'H1': [1]})
        atoms = [['H0', ( 0, 0, 0)],
                 ['H1', ( 0, 0, 0)],
                 ['H2', ( 0, 0, 0)],
                 ['H3', ( 0, 0, 0)]]
        basis = {'H2':'sto3g', 'H3':'6-31g', 'H0':'sto3g', 'H1': '6-31g'}
        atmgroup = gto.mole.atom_types(atoms, basis)
        self.assertEqual(atmgroup, {'H2': [2], 'H3': [3], 'H0': [0], 'H1': [1]})

    def test_input_symmetry(self):
        mol = gto.M(atom='H 0 0 -1; H 0 0 1', symmetry='D2h')
        self.assertEqual(mol.irrep_id, [0, 5])
        mol = gto.M(atom='H 0 0 -1; H 0 0 1', symmetry='D2')
        self.assertEqual(mol.irrep_id, [0, 1])
        mol = gto.M(atom='H 0 0 -1; H 0 0 1', symmetry='C2v')
        self.assertEqual(mol.irrep_id, [0])

    def test_dumps_loads(self):
        import warnings
        mol1 = gto.M()
        mol1.x = lambda *args: None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = mol1.dumps()
            self.assertTrue(w[0].category, UserWarning)
        mol1.loads(mol0.dumps())

    def test_symm_orb_serialization(self):
        '''Handle the complex symmetry-adapted orbitals'''
        mol = gto.M(atom='He', basis='ccpvdz', symmetry=True)
        mol.loads(mol.dumps())

        lz_minus = numpy.sqrt(.5) * (mol.symm_orb[3] - mol.symm_orb[2] * 1j)
        lz_plus = -numpy.sqrt(.5) * (mol.symm_orb[3] + mol.symm_orb[2] * 1j)
        mol.symm_orb[2] = lz_minus
        mol.symm_orb[3] = lz_plus
        mol.loads(mol.dumps())
        self.assertTrue(mol.symm_orb[0].dtype == numpy.double)
        self.assertTrue(mol.symm_orb[2].dtype == numpy.complex128)
        self.assertTrue(mol.symm_orb[3].dtype == numpy.complex128)

    def test_same_mol1(self):
        self.assertTrue(gto.same_mol(mol0, mol0))
        mol1 = gto.M(atom='h   0  1  1; O1  0  0  0;  h   1  1  0')
        self.assertTrue(not gto.same_mol(mol0, mol1))
        self.assertTrue(gto.same_mol(mol0, mol1, cmp_basis=False))

        mol1 = gto.M(atom='h   0  1  1; O1  0  0  0;  h   1  1  0.01')
        self.assertTrue(not gto.same_mol(mol0, mol1, cmp_basis=False))
        self.assertTrue(gto.same_mol(mol0, mol1, tol=.02, cmp_basis=False))

        mol1 = gto.M(atom='''H 0.0052917700 0.0000000000 -0.8746076326
                             F 0.0000000000 0.0000000000 0.0516931447''')
        mol2 = gto.M(atom='''H 0.0000000000 0.0000000000 -0.8746076326
                             F 0.0000000000 0.0000000000 0.0516931447''')
        self.assertTrue(gto.same_mol(mol1, mol2))
        self.assertTrue(not gto.same_mol(mol1, mol2, tol=1e-6))
        mol3 = gto.M(atom='''H 0.0000000000 0.0000000000 -0.8746076326
                             H 0.0000000000 0.0000000000 0.0516931447''')
        self.assertTrue(not gto.same_mol(mol3, mol2))

    def test_same_mol2(self):
        mol1 = gto.M(atom='H 0.0052917700 0.0000000000 -0.8746076326; F 0.0000000000 0.0000000000 0.0464013747')
        mol2 = gto.M(atom='H 0.0000000000 0.0000000000 -0.8746076326; F 0.0052917700 0.0000000000 0.0464013747')
        self.assertTrue(gto.same_mol(mol1, mol2))

        mol1 = gto.M(atom='H 0.0052917700 0.0000000000 -0.8693158626; F 0.0000000000 0.0000000000 0.0464013747')
        mol2 = gto.M(atom='H 0.0000000000 0.0052917700 -0.8693158626; F 0.0000000000 0.0000000000 0.0464013747')
        mol3 = gto.M(atom='H 0.0000000000 0.0000000000 -0.8693158626; F 0.0052917700 0.0000000000 0.0464013747')
        mol4 = gto.M(atom='H -0.0052917700 0.0000000000 -0.8746076326; F 0.0000000000 0.0000000000 0.0411096047')
        mols = (mol1, mol2, mol3, mol4)
        for i,mi in enumerate(mols):
            for j in range(i):
                self.assertTrue(gto.same_mol(mols[i], mols[j]))

        mol1 = gto.M(atom='''H 0.0000000000 0.0000000000 0.0000000000
          H 0.9497795800 1.3265673200 0.0000000000
          H 0.9444878100 -1.3265673200 0.0000000000
          H1 -0.9444878100 0.0000000000 1.3265673200
          H1 -0.9444878100 0.0000000000 -1.3265673200''', basis={'H':'sto3g', 'H1':'sto3g'}, charge=1)
        mol2 = gto.M(atom='''H 0.0000000000 0.0000000000 0.0000000000
          H 0.9444878100 1.3265673200 0.0000000000
          H 0.9497795800 -1.3265673200 0.0000000000
          H1 -0.9444878100 0.0000000000 1.3265673200
          H1 -0.9444878100 0.0000000000 -1.3265673200''', basis={'H':'sto3g', 'H1':'sto3g'}, charge=1)
        self.assertTrue(gto.same_mol(mol1, mol2))
        self.assertEqual(len(gto.atom_types(mol1._atom)), 2)
        mol3 = gto.M(atom='''H 0.0000000000 0.0000000000 0.0000000000
          H1 0.9497795800 1.3265673200 0.0000000000
          H1 0.9444878100 -1.3265673200 0.0000000000
          H1 -0.9444878100 0.0000000000 1.3265673200
          H1 -0.9444878100 0.0000000000 -1.3265673200''', basis={'H':'sto3g', 'H1':'321g'}, charge=1)
        self.assertTrue(not gto.same_mol(mol3, mol2))

    def test_inertia_momentum(self):
        mol1 = gto.Mole()
        mol1.atom = mol0.atom
        mol1.nucmod = 'G'
        mol1.verbose = 5
        mol1.nucprop = {'H': {'mass': 3}}
        mol1.output = '/dev/null'
        mol1.build(False, False)
        self.assertAlmostEqual(lib.fp(gto.inertia_moment(mol1)),
                               5.340587366981696, 9)

        mass = mol0.atom_mass_list(isotope_avg=True)
        self.assertAlmostEqual(lib.fp(gto.inertia_moment(mol1, mass)),
                               2.1549269955776205, 9)

    def test_chiral_mol(self):
        mol1 = gto.M(atom='C 0 0 0; H 1 1 1; He -1 -1 1; Li -1 1 -1; Be 1 -1 -1')
        mol2 = gto.M(atom='C 0 0 0; H 1 1 1; He -1 -1 1; Be -1 1 -1; Li 1 -1 -1')
        self.assertTrue(gto.chiral_mol(mol1, mol2))
        self.assertTrue(gto.chiral_mol(mol1))

        mol1 = gto.M(atom='''H 0.9444878100 1.3265673200 0.0052917700
                            H 0.9444878100 -1.3265673200 0.0000000000
                            H -0.9444878100 0.0000000000 1.3265673200
                            H -0.9444878100 0.0000000000 -1.3265673200''')
        mol2 = gto.M(atom='''H 0.9444878100 1.3265673200 0.0000000000
                            H 0.9444878100 -1.3265673200 0.0052917700
                            H -0.9444878100 0.0000000000 1.3265673200
                            H -0.9444878100 0.0000000000 -1.3265673200''')
        self.assertTrue(gto.chiral_mol(mol1, mol2))

        mol1 = gto.M(atom='''H 0.9444878100 1.3265673200 0.0052917700
                            H 0.9444878100 -1.3265673200 0.0000000000
                            H -0.9444878100 0.0000000000 1.3265673200
                            H -0.9444878100 0.0000000000 -1.3265673200''')
        self.assertTrue(gto.chiral_mol(mol1))

    def test_first_argument(self):
        mol1 = gto.Mole()
        mol1.build('He')
        self.assertEqual(mol1.natm, 1)

    def test_atom_as_file(self):
        ftmp = tempfile.NamedTemporaryFile('w')
        # file in xyz format
        ftmp.write('He 0 0 0\nHe 0 0 1\n')
        ftmp.flush()
        mol1 = gto.M(atom=ftmp.name)
        self.assertEqual(mol1.natm, 2)

        # file in zmatrix format
        ftmp = tempfile.NamedTemporaryFile('w')
        ftmp.write('He\nHe 1 1.5\n')
        ftmp.flush()
        mol1 = gto.M(atom=ftmp.name)
        self.assertEqual(mol1.natm, 2)

    def test_format_atom(self):
        atoms = [['h' , 0,1,1], "O1  0. 0. 0.", [1, 1.,1.,0.],]
        self.assertTrue(numpy.allclose(gto.mole.format_atom(atoms, unit='Ang')[0][1],
                                       [0.0, 1.8897261245650618, 1.8897261245650618]))
        atoms = '''h 0 1 1
        O1 0 0 0; 1 1 1 0; #H 0 0 3'''
        self.assertTrue(numpy.allclose(gto.mole.format_atom(atoms, unit=1)[0][1],
                                       [0.0, 1., 1.]))
        atoms = 'O1; h 1 1; 1 1 1 2 90'
        atoms = gto.mole.format_atom(atoms, unit=1)[2]
        self.assertEqual(atoms[0], 'H')
        self.assertTrue(numpy.allclose(atoms[1], [0, 0, 1.]))

    def test_format_basis(self):
        mol = gto.M(atom = '''O 0 0 0; 1 0 1 0; H 0 0 1''',
                    basis = {8: 'ccpvdz'})
        self.assertEqual(mol.nao_nr(), 14)

        mol = gto.M(atom = '''O 0 0 0; H:1 0 1 0; H@2 0 0 1''',
                    basis = {'O': 'ccpvdz', 'H:1': 'sto3g', 'H': 'unc-iglo3'})
        self.assertEqual(mol.nao_nr(), 32)

        mol = gto.M(
            atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
            basis = {'default': ('6-31g', [[0, [.05, 1.]], []]), 'H2': 'sto3g'}
        )
        self.assertEqual(mol.nao_nr(), 14)

        mol = gto.M(
            atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
            basis = {'H1': gto.parse('''
# Parse NWChem format basis string (see https://bse.pnl.gov/bse/portal).
# Comment lines are ignored
#BASIS SET: (6s,3p) -> [2s,1p]
        H    S
              2.9412494             -0.09996723
              0.6834831              0.39951283
              0.2222899              0.70011547
        H    S
              2.9412494             0.15591627
              0.6834831             0.60768372
              0.2222899             0.39195739
                                    ''', optimize=True),
                     'O': 'unc-ccpvdz',
                     'H2': gto.load('sto-3g', 'He')  # or use basis of another atom
                    }
        )
        self.assertEqual(mol.nao_nr(), 29)

        mol = gto.M(
            atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
            basis = {'H': ['sto3g', '''unc
        C    S
             71.6168370              0.15432897
             13.0450960              0.53532814
              3.5305122              0.44463454
        C    SP
              2.9412494             -0.09996723             0.15591627
              0.6834831              0.39951283             0.60768372
              0.2222899              0.70011547             0.39195739
        '''],
                     'O': mol.expand_etbs([(0, 4, 1.5, 2.2),  # s-function
                                           (1, 2, 0.5, 2.2)]) # p-function
                    }
        )
        self.assertEqual(mol.nao_nr(), 42)

        mol = gto.M(
            atom = '''O 0 0 0; H1 0 1 0; H2 0 0 1''',
            basis = ('sto3g', 'ccpvdz', '3-21g',
                     gto.etbs([(0, 4, 1.5, 2.2), (1, 2, 0.5, 2.2)]),
                    [[0, numpy.array([1e3, 1.])]])
        )
        self.assertEqual(mol.nao_nr(), 77)

        mol.atom = 'Hg'
        mol.basis = 'ccpvdz'
        self.assertRaises(RuntimeError, mol.build)

    def test_default_basis(self):
        mol = gto.M(atom=[['h' , 0,1,1], ["O1", (0.,0.,0.)], [1, 1.,1.,0.],],
                    basis={'default':'321g', 'O1': 'sto3g'})
        self.assertEqual(sorted(mol._basis.keys()), ['H', 'O1'])

    def test_parse_pople_basis(self):
        self.assertEqual(len(gto.basis.load('6-31G(d)'      , 'H')), 2)
        self.assertEqual(len(gto.basis.load('6-31G(d)'      , 'C')), 6)
        self.assertEqual(len(gto.basis.load('6-31Gs'        , 'C')), 6)
        self.assertEqual(len(gto.basis.load('6-31G*'        , 'C')), 6)
        self.assertEqual(len(gto.basis.load('6-31G(d,p)'    , 'H')), 3)
        self.assertEqual(len(gto.basis.load('6-31G(d,p)'    , 'C')), 6)
        self.assertEqual(len(gto.basis.load('6-31G(2d,2p)'  , 'H')), 4)
        self.assertEqual(len(gto.basis.load('6-31G(2d,2p)'  , 'C')), 7)
        self.assertEqual(len(gto.basis.load('6-31G(3df,3pd)', 'H')), 6)
        self.assertEqual(len(gto.basis.load('6-31G(3df,3pd)', 'C')), 9)

    def test_parse_basis(self):
        mol = gto.M(atom='''
                    6        0    0   -0.5
                    8        0    0    0.5
                    1        1    0   -1.0
                    1       -1    0   -1.0''',
                    basis='''
#BASIS SET: (3s) -> [2s]
H    S
      5.4471780              0.1562849787        
      0.82454724             0.9046908767        
H    S
      0.18319158             1.0000000        
#BASIS SET: (6s,3p) -> [3s,2p]
C    S
    172.2560000              0.0617669        
     25.9109000              0.3587940        
      5.5333500              0.7007130        
C    SP
      3.6649800             -0.3958970              0.2364600        
      0.7705450              1.2158400              0.8606190        
C    SP
      0.1958570              1.0000000              1.0000000        
#BASIS SET: (6s,3p) -> [3s,2p]
O    S
    322.0370000              0.0592394        
     48.4308000              0.3515000        
     10.4206000              0.7076580        
O    SP
      7.4029400             -0.4044530              0.2445860        
      1.5762000              1.2215600              0.8539550        
O    SP
      0.3736840              1.0000000              1.0000000        
''')
        self.assertTrue(mol.nao_nr() == 22)

    def test_ghost(self):
        mol = gto.M(
            atom = 'C 0 0 0; ghost 0 0 2',
            basis = {'C': 'sto3g', 'ghost': gto.basis.load('sto3g', 'H')}
        )
        self.assertEqual(mol.nao_nr(), 6)

        mol = gto.M(atom='''
        ghost-O     0.000000000     0.000000000     2.500000000
        X_H        -0.663641000    -0.383071000     3.095377000
        ghost.H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''',
        basis='631g')
        self.assertEqual(mol.nao_nr(), 26)

        mol = gto.M(atom='''
        ghost-O     0.000000000     0.000000000     2.500000000
        X_H        -0.663641000    -0.383071000     3.095377000
        ghost.H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        ''',
                basis={'H': '3-21g', 'o': '3-21g', 'ghost-O': 'sto3g'})
        self.assertEqual(mol.nao_nr(), 18) # 5 + 2 + 2 + 9

        mol = gto.M(atom='Zn 0 0 0; ghost-Fe 0 0 1',
                    basis='lanl2dz', ecp='lanl2dz')
        self.assertTrue(len(mol._ecp) == 1)  # only Zn ecp

        mol = gto.M(atom='Zn 0 0 0; ghost-Fe 0 0 1',
                    basis='lanl2dz', ecp={'Zn': 'lanl2dz', 'ghost-Fe': 'lanl2dz'})
        self.assertTrue(len(mol._ecp) == 2)  # Zn and ghost-Fe in ecp

    def test_nucmod(self):
        gto.filatov_nuc_mod(80)
        self.assertEqual(gto.mole._parse_nuc_mod(1), gto.NUC_GAUSS)
        self.assertEqual(gto.mole._parse_nuc_mod('Gaussian'), gto.NUC_GAUSS)
        mol1 = gto.Mole()
        mol1.atom = mol0.atom
        mol1.nucmod = 'G'
        mol1.verbose = 5
        mol1.nucprop = {'H': {'mass': 3}}
        mol1.output = '/dev/null'
        mol1.build(False, False)
        mol1.set_nuc_mod(0, 2)
        self.assertTrue(mol1._atm[1,gto.NUC_MOD_OF] == gto.NUC_GAUSS)
        self.assertAlmostEqual(mol1._env[mol1._atm[0,gto.PTR_ZETA]], 2, 9)
        self.assertAlmostEqual(mol1._env[mol1._atm[1,gto.PTR_ZETA]], 586314366.54656982, 4)

        mol1.set_nuc_mod(1, 0)
        self.assertTrue(mol1._atm[1,gto.NUC_MOD_OF] == gto.NUC_POINT)

        mol1.nucmod = None
        mol1.build(False, False)
        self.assertTrue(mol1._atm[1,gto.NUC_MOD_OF] == gto.NUC_POINT)

        mol1.nucmod = {'H': gto.filatov_nuc_mod}
        mol1.build(False, False)
        self.assertTrue(mol1._atm[0,gto.NUC_MOD_OF] == gto.NUC_GAUSS)
        self.assertTrue(mol1._atm[1,gto.NUC_MOD_OF] == gto.NUC_POINT)
        self.assertTrue(mol1._atm[2,gto.NUC_MOD_OF] == gto.NUC_GAUSS)

    def test_zmat(self):
        coord = numpy.array((
            (0.200000000000, -1.889726124565,  0.000000000000),
            (1.300000000000, -1.889726124565,  0.000000000000),
            (2.400000000000, -1.889726124565,  0.000000000000),
            (3.500000000000, -1.889726124565,  0.000000000000),
            (0.000000000000,  0.000000000000, -1.889726124565),
            (0.000000000000,  1.889726124565,  0.000000000000),
            (0.200000000000, -0.800000000000,  0.000000000000),
            (1.889726124565,  0.000000000000,  1.133835674739)))
        zstr0 = gto.cart2zmat(coord)
        zstr = '\n'.join(['H '+x for x in zstr0.splitlines()])
        atoms = gto.zmat2cart(zstr)
        zstr1 = gto.cart2zmat([x[1] for x in atoms])
        self.assertTrue(zstr0 == zstr1)

        numpy.random.seed(1)
        coord = numpy.random.random((6,3))
        zstr0 = gto.cart2zmat(coord)
        zstr = '\n'.join(['H '+x for x in zstr0.splitlines()])
        atoms = gto.zmat2cart(zstr)
        zstr1 = gto.cart2zmat([x[1] for x in atoms])
        self.assertTrue(zstr0 == zstr1)

    def test_c2s(self):  # Transformation of cart <-> sph, sph <-> spinor
        c = mol0.sph2spinor_coeff()
        s0 = mol0.intor('int1e_ovlp_spinor')
        s1 = mol0.intor('int1e_ovlp_sph')
        sa = reduce(numpy.dot, (c[0].T.conj(), s1, c[0]))
        sa+= reduce(numpy.dot, (c[1].T.conj(), s1, c[1]))
        mol0.cart = True
        s2 = mol0.intor('int1e_ovlp')
        mol0.cart = False
        self.assertAlmostEqual(abs(s0 - sa).max(), 0, 12)
        c = mol0.cart2sph_coeff()
        sa = reduce(numpy.dot, (c.T.conj(), s2, c))
        self.assertAlmostEqual(abs(s1 - sa).max(), 0, 12)

        c0 = gto.mole.cart2sph(1)
        ca, cb = gto.mole.cart2spinor_l(1)
        ua, ub = gto.mole.sph2spinor_l(1)
        self.assertAlmostEqual(abs(c0.dot(ua)-ca).max(), 0, 9)
        self.assertAlmostEqual(abs(c0.dot(ub)-cb).max(), 0, 9)

        c0 = gto.mole.cart2sph(0, normalized='sp')
        ca, cb = gto.mole.cart2spinor_kappa(-1, 0, normalized='sp')
        ua, ub = gto.mole.sph2spinor_kappa(-1, 0)
        self.assertAlmostEqual(abs(c0.dot(ua)-ca).max(), 0, 9)
        self.assertAlmostEqual(abs(c0.dot(ub)-cb).max(), 0, 9)

        c1 = gto.mole.cart2sph(0, numpy.eye(1))
        self.assertAlmostEqual(abs(c0*0.282094791773878143-c1).max(), 0, 12)

        c0 = gto.mole.cart2sph(1, normalized='sp')
        ca, cb = gto.mole.cart2spinor_kappa(1, 1, normalized='sp')
        ua, ub = gto.mole.sph2spinor_kappa(1, 1)
        self.assertAlmostEqual(abs(c0.dot(ua)-ca).max(), 0, 9)
        self.assertAlmostEqual(abs(c0.dot(ub)-cb).max(), 0, 9)

        c1 = gto.mole.cart2sph(1, numpy.eye(3).T)
        self.assertAlmostEqual(abs(c0*0.488602511902919921-c1).max(), 0, 12)

    def test_bas_method(self):
        self.assertEqual([mol0.bas_len_cart(x) for x in range(mol0.nbas)],
                         [1, 3, 1, 1, 1, 1, 1, 3, 3, 3, 6, 6, 1, 3])
        self.assertEqual([mol0.bas_len_spinor(x) for x in range(mol0.nbas)],
                         [2, 4, 2, 2, 2, 2, 2, 6, 6, 6, 10, 10, 2, 4])
        c0 = mol0.bas_ctr_coeff(0)
        self.assertAlmostEqual(abs(c0[:,0]/c0[0,0] - (1,3,1)).max(), 0, 9)
        self.assertAlmostEqual(abs(c0[:,1] - (0,1,0)).max(), 0, 9)

        self.assertRaises(ValueError, mol0.gto_norm, -1, 1.)

    def test_nelectron(self):
        mol = gto.Mole()
        mol.atom = [
            [1  , (0.,1.,1.)],
            ["O1", (0.,0.,0.)],
            [1  , (1.,1.,0.)], ]
        mol.charge = 1
        self.assertEqual(mol.nelectron, 9)

        mol0.nelectron = mol0.nelectron
        mol0.nelectron = mol0.nelectron
        mol0.spin = 2
        self.assertRaises(RuntimeError, lambda *args: mol0.nelec)
        mol0.spin = 1

        mol1 = mol0.copy()
        self.assertEqual(mol1.nelec, (5, 4))
        mol1.nelec = (3, 6)
        self.assertEqual(mol1.nelec, (3, 6))

    def test_multiplicity(self):
        mol1 = mol0.copy()
        self.assertEqual(mol1.multiplicity, 2)
        mol1.multiplicity = 5
        self.assertEqual(mol1.multiplicity, 5)
        self.assertEqual(mol1.spin, 4)
        self.assertRaises(RuntimeError, lambda:mol1.nelec)

    def test_ms(self):
        mol1 = mol0.copy()
        self.assertEqual(mol1.ms, 0.5)
        mol1.ms = 1
        self.assertEqual(mol1.multiplicity, 3)
        self.assertEqual(mol1.spin, 2)
        self.assertRaises(RuntimeError, lambda:mol1.nelec)

    def test_basis_not_found(self):
        mol = gto.M(atom='''
        H     -0.663641000    -0.383071000     3.095377000
        H     0.663588000     0.383072000     3.095377000
        O     0.000000000     0.000000000     2.500000000
        H     -0.663641000    -0.383071000     3.095377000
        H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H     -0.663641000    -0.383071000     3.095377000
        H     0.663588000     0.383072000     3.095377000
        ''', basis={'O': '3-21g'})
        #TODO: assert the warning "Warn: Basis not found for atom 1 H"
        self.assertEqual(mol.nao_nr(), 18)

        aoslice = mol.aoslice_by_atom()
        self.assertEqual(aoslice[:,0].tolist(), [0, 0, 0, 5, 5, 5,10,10])
        self.assertEqual(aoslice[:,1].tolist(), [0, 0, 5, 5, 5,10,10,10])

    def test_atom_method(self):
        aoslice = mol0.aoslice_by_atom()
        for i in range(mol0.natm):
            symb = mol0.atom_pure_symbol(i)
            shls = mol0.atom_shell_ids(i)
            nshls = aoslice[i][1] - aoslice[i][0]
            self.assertEqual(shls[0], aoslice[i][0])
            self.assertEqual(len(shls), nshls)
            self.assertEqual(mol0.atom_nshells(i), nshls)
        aoslice = mol0.aoslice_2c_by_atom()
        mol0.elements  # test property(elements) in Mole
        self.assertEqual([x[2] for x in aoslice], [0, 8, 56])
        self.assertEqual([x[3] for x in aoslice], [8, 56, 64])

    def test_dump_loads_skip(self):
        import json
        with tempfile.NamedTemporaryFile() as tmpfile:
            lib.chkfile.save_mol(mol0, tmpfile.name)
            mol1 = gto.Mole()
            mol1.update(tmpfile.name)
            # dumps() may produce different orders in different runs
            self.assertEqual(json.loads(mol1.dumps()), json.loads(mol0.dumps()))
        mol1.loads(mol1.dumps())
        mol1.loads_(mol0.dumps())
        mol1.unpack(mol1.pack())
        mol1.unpack_(mol0.pack())

    def test_set_geom(self):
        mol1 = gto.Mole()
        mol1.verbose = 5
        mol1.set_geom_(mol0._atom, 'B', symmetry=True)
        mol1.set_geom_(mol0.atom_coords(), 'B', inplace=False)

        mol1.symmetry = False
        mol1.set_geom_(mol0.atom_coords(), 'B')
        mol1.set_geom_(mol0.atom_coords(), inplace=False)
        mol1.set_geom_(mol0.atom_coords(), unit=1.)
        mol1.set_geom_(mol0.atom_coords(), unit='Ang', inplace=False)

    def test_apply(self):
        from pyscf import scf, mp
        self.assertTrue(isinstance(mol0.apply('RHF'), scf.rohf.ROHF))
        self.assertTrue(isinstance(mol0.apply('MP2'), mp.ump2.UMP2))
        self.assertTrue(isinstance(mol0.apply(scf.RHF), scf.rohf.ROHF))
        self.assertTrue(isinstance(mol0.apply(scf.uhf.UHF), scf.uhf.UHF))

    def test_with_MoleContext(self):
        mol1 = mol0.copy()
        with mol1.with_rinv_at_nucleus(1):
            self.assertTrue(mol1._env[gto.PTR_RINV_ZETA] != 0)
            self.assertAlmostEqual(abs(mol1._env[gto.PTR_RINV_ORIG+2]), 0, 9)
        self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ZETA], 0, 9)
        self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ORIG+2], 0, 9)
        with mol1.with_rinv_at_nucleus(0):
            self.assertAlmostEqual(abs(mol1._env[gto.PTR_RINV_ORIG+2]), 1.8897261245650618, 9)
        self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ORIG+2], 0, 9)

        with mol1.with_rinv_zeta(20):
            self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ZETA], 20, 9)
            mol1.set_rinv_zeta(3.)
        self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ZETA], 0, 9)

        with mol1.with_rinv_origin((1,2,3)):
            self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ORIG+2], 3, 9)
        self.assertAlmostEqual(mol1._env[gto.PTR_RINV_ORIG+2], 0, 9)

        with mol1.with_range_coulomb(20):
            self.assertAlmostEqual(mol1._env[gto.PTR_RANGE_OMEGA], 20, 9)
            mol1.set_range_coulomb(2.)
        self.assertAlmostEqual(mol1._env[gto.PTR_RANGE_OMEGA], 0, 9)

        with mol1.with_common_origin((1,2,3)):
            self.assertAlmostEqual(mol1._env[gto.PTR_COMMON_ORIG+2], 3, 9)
        self.assertAlmostEqual(mol1._env[gto.PTR_COMMON_ORIG+2], 0, 9)

        mol1.set_f12_zeta(2.)

    def test_input_symmetry1(self):
        mol1 = gto.Mole()
        mol1.atom = 'H 1 1 1; H -1 -1 1; H 1 -1 -1; H -1 1 -1'
        mol1.unit = 'B'
        mol1.symmetry = True
        mol1.verbose = 5
        mol1.output = '/dev/null'
        mol1.build()
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), 3.4708548731841296, 9)

        mol1 = gto.Mole()
        mol1.atom = 'H 0 0 -1; H 0 0 1'
        mol1.cart = True
        mol1.unit = 'B'
        mol1.symmetry = 'Dooh'
        mol1.verbose = 5
        mol1.output = '/dev/null'
        mol1.build()
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), 0.69980902201036865, 9)

        mol1 = gto.Mole()
        mol1.atom = 'H 0 -1 0; H 0 1 0'
        mol1.unit = 'B'
        mol1.symmetry = True
        mol1.symmetry_subgroup = 'D2h'
        mol1.build()
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), -1.1939459267317516, 9)

        mol1.atom = 'H 0 0 -1; H 0 0 1'
        mol1.unit = 'B'
        mol1.symmetry = 'Coov'
        mol1.symmetry_subgroup = 'C2'
        mol1.build()
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), 0.69980902201036865, 9)

        mol1.atom = 'H 1 0 -1; H 0 0 1; He 0 0 2'
        mol1.symmetry = 'Coov'
        self.assertRaises(PointGroupSymmetryError, mol1.build)

        mol1.atom = '''
        C 0. 0. 0.7264
        C 0. 0. -.7264
        H 0.92419 0. 1.29252
        H -.92419 0. 1.29252
        H 0. 0.92419 -1.29252
        H 0. -.92419 -1.29252'''
        mol1.symmetry = True
        mol1.symmetry_subgroup = 'C2v'
        mol1.build()
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), 2.9413856643164618, 9)

        mol1 = gto.Mole()
        mol1.atom = [
            ["O" , (0. , 0.     , 0.)],
            ["H" , (0. , -0.757 , 0.587)],
            ["H" , (0. , 0.757  , 0.587)]]
        mol1.symmetry = "C3"
        self.assertRaises(PointGroupSymmetryError, mol1.build)

        mol1 = gto.Mole()
        mol1.atom = 'H 0 0 0; H 1 0 0'
        mol1.basis = 'sto-3g'
        mol1.symmetry = 'Dooh'
        mol1.build()
        self.assertAlmostEqual(abs(mol1._symm_axes - numpy.eye(3)[[1,2,0]]).max(), 0, 9)

        mol1 = gto.M(
            atom='He 0 0 0',
            basis='aug-cc-pvdz',
            symmetry='SO3'
        )
        self.assertEqual(mol1.groupname, 'SO3')

    def test_symm_orb(self):
        rs = numpy.array([[.1, -.3, -.2],
                          [.3,  .1,  .8]])
        mol = gto.M(atom=[('H', c) for c in rs], unit='Bohr',
                    basis={'H': [[0, (1, 1)], [1, (.9, 1)], [2, (.8, 1)], [3, (.7, 1)]]})

        numpy.random.seed(1)
        u, w, vh = numpy.linalg.svd(numpy.random.random((3,3)))
        rs1 = rs.dot(u) + numpy.array([-.5, -.3, .9])
        mol1 = gto.M(atom=[('H', c) for c in rs1], unit='Bohr',
                     basis={'H': [[0, (1, 1)], [1, (.9, 1)], [2, (.8, 1)], [3, (.7, 1)]]})

        mol.symmetry = 1
        mol.build()
        mol1.symmetry = 1
        mol1.build()

        s0 = mol.intor('int1e_ovlp')
        s0 = [abs(c.T.dot(s0).dot(c)) for c in mol.symm_orb]
        s1 = mol1.intor('int1e_ovlp')
        s1 = [abs(c.T.dot(s1).dot(c)) for c in mol1.symm_orb]
        self.assertTrue(all(abs(s0[i]-s1[i]).max()<1e-12 for i in range(len(mol.symm_orb))))

        mol.cart = True
        mol.symmetry = 1
        mol.build()
        mol1.cart = True
        mol1.symmetry = 1
        mol1.build()

        s0 = mol.intor('int1e_ovlp')
        s0 = [abs(c.T.dot(s0).dot(c)) for c in mol.symm_orb]
        s1 = mol1.intor('int1e_ovlp')
        s1 = [abs(c.T.dot(s1).dot(c)) for c in mol1.symm_orb]
        self.assertTrue(all(abs(s0[i]-s1[i]).max()<1e-12 for i in range(len(mol.symm_orb))))

    def test_search_ao_label(self):
        mol1 = mol0.copy()
        mol1.atom = mol0.atom + ['Mg 1,1,1']
        mol1.ecp['Mg'] = 'lanl2dz'
        mol1.basis['Mg'] = 'lanl2dz'
        mol1.build(0, 0)
        self.assertEqual(list(mol1.search_ao_label('O.*2p')), [10,11,12])
        self.assertEqual(list(mol1.search_ao_label('O1 2p')), [10,11,12])
        self.assertEqual(list(mol1.search_ao_label(['O.*2p','0 H 1s'])), [0, 10,11,12])
        self.assertEqual(list(mol1.search_ao_label([10,11,12])), [10,11,12])
        self.assertEqual(list(mol1.search_ao_label(lambda x: '4d' in x)), [24,25,26,27,28])
        mol1.ao_labels(fmt='%s%s%s%s')
        mol1.sph_labels(fmt=None)
        mol1.cart = True
        self.assertEqual(list(mol1.search_ao_label('4d')), [25,26,27,28,29,30])
        mol1.ao_labels(fmt='%s%s%s%s')
        mol1.ao_labels(fmt=None)
        mol1.cart = False
        mol1.spinor_labels()
        mol1.spinor_labels(fmt='%s%s%s%s')
        mol1.spinor_labels(fmt=None)

    def test_input_ecp(self):
        mol1 = gto.Mole()
        mol1.atom = mol0.atom
        mol1.ecp = 'lanl2dz'
        mol1.build(False, False)
        gto.basis.load_ecp('lanl08', 'O')
        gto.format_ecp({'O':'lanl08', 1:'lanl2dz'})
        self.assertRaises(BasisNotFoundError, gto.format_ecp, {'H':'lan2ldz'})

    def test_condense_to_shell(self):
        mol1 = mol0.copy()
        mol1.symmetry = False
        mol1.build(False, False)
        v = gto.condense_to_shell(mol1, mol1.intor('int1e_ovlp'), numpy.max)
        self.assertAlmostEqual(lib.fp(v), 5.7342530154117846, 9)

    def test_input_ghost_atom(self):
        mol = gto.M(
            atom = 'C 0 0 0; ghost 0 0 2',
            basis = {'C': 'sto3g', 'ghost': gto.basis.load('sto3g', 'H')}
        )

        mol = gto.M(atom='''
        ghost1     0.000000000     0.000000000     2.500000000
        ghost2    -0.663641000    -0.383071000     3.095377000
        ghost2     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''',
                    basis={'ghost1':gto.basis.load('sto3g', 'O'),
                           'ghost2':gto.basis.load('631g', 'H'),
                           'O':'631g', 'H':'631g'}
        )

        mol = gto.M(atom='''
        ghost-O     0.000000000     0.000000000     2.500000000
        ghost_H    -0.663641000    -0.383071000     3.095377000
        ghost:H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''', basis='631g')

        mol = gto.M(atom='''
        X1     0.000000000     0.000000000     2.500000000
        X2    -0.663641000    -0.383071000     3.095377000
        X2     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''',
                    basis={'X1':gto.basis.load('sto3g', 'O'),
                           'X2':gto.basis.load('631g', 'H'),
                           'O':'631g', 'H':'631g'}
        )

        mol = gto.M(atom='''
        X-O     0.000000000     0.000000000     2.500000000
        X_H1   -0.663641000    -0.383071000     3.095377000
        X:H     0.663588000     0.383072000     3.095377000
        O     1.000000000     0.000000000     2.500000000
        H    -1.663641000    -0.383071000     3.095377000
        H     1.663588000     0.383072000     3.095377000
        ''', basis='631g')

    def test_conc_mole(self):
        mol1 = gto.M(atom='Mg', ecp='LANL2DZ', basis='lanl2dz')
        mol2 = mol1 + mol0
        self.assertEqual(mol2.natm, 4)
        self.assertEqual(mol2.nbas, 18)
        self.assertEqual(mol2.nao_nr(), 42)
        mol2 = mol0 + mol1
        self.assertEqual(mol2.natm, 4)
        self.assertEqual(mol2.nbas, 18)
        self.assertEqual(mol2.nao_nr(), 42)
        n0 = mol0.npgto_nr()
        n1 = mol1.npgto_nr()
        self.assertEqual(mol2.npgto_nr(), n0+n1)
        mol2 = mol2 + mol2
        mol2.cart = True
        self.assertEqual(mol2.npgto_nr(), 100)
        mol3 = gto.M(atom='Cu', basis='lanl2dz', ecp='lanl2dz', spin=None)
        mol4 = mol1 + mol3
        self.assertEqual(len(mol4._ecpbas), 16)

    def test_intor_cross_cart(self):
        mol1 = gto.M(atom='He', basis={'He': [(2,(1.,1))]}, cart=True)
        s0 = gto.intor_cross('int1e_ovlp', mol1, mol0)
        self.assertEqual(s0.shape, (6, 34))
        s0 = gto.intor_cross('int1e_ovlp', mol0, mol1)
        self.assertEqual(s0.shape, (34, 6))
        s0 = gto.intor_cross('int1e_ovlp_cart', mol0, mol1)
        self.assertEqual(s0.shape, (36, 6))

    def test_energy_nuc(self):
        self.assertAlmostEqual(mol0.get_enuc(), 6.3611415029455705, 9)
        self.assertAlmostEqual(gto.M().energy_nuc(), 0, 9)

    def test_fakemol(self):
        numpy.random.seed(1)
        coords = numpy.random.random((6,3))*4
        vref = 0
        mol = mol0.copy()
        for c in coords:
            mol.set_rinv_origin(c)
            vref += mol.intor('int1e_rinv')

        fakemol = gto.fakemol_for_charges(coords)
        pmol = mol + fakemol
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas)
        v = pmol.intor('int3c2e', comp=1, shls_slice=shls_slice)
        v = numpy.einsum('pqk->pq', v)
        self.assertAlmostEqual(abs(vref-v).max(), 0, 12)

    def test_to_uncontracted_cartesian_basis(self):
        pmol, ctr_coeff = mol0.to_uncontracted_cartesian_basis()
        c = scipy.linalg.block_diag(*ctr_coeff)
        s = reduce(numpy.dot, (c.T, pmol.intor('int1e_ovlp'), c))
        self.assertAlmostEqual(abs(s-mol0.intor('int1e_ovlp')).max(), 0, 9)

        with lib.temporary_env(mol0, cart=True):
            pmol, ctr_coeff = mol0.to_uncontracted_cartesian_basis()
            c = scipy.linalg.block_diag(*ctr_coeff)
            s = reduce(numpy.dot, (c.T, pmol.intor('int1e_ovlp'), c))
            self.assertAlmostEqual(abs(s-mol0.intor('int1e_ovlp')).max(), 0, 9)

    def test_getattr(self):
        from pyscf import scf, dft, ci, tdscf
        mol = gto.M(atom='He')
        self.assertEqual(mol.HF().__class__, scf.HF(mol).__class__)
        self.assertEqual(mol.KS().__class__, dft.KS(mol).__class__)
        self.assertEqual(mol.UKS().__class__, dft.UKS(mol).__class__)
        self.assertEqual(mol.CISD().__class__, ci.cisd.RCISD)
        self.assertEqual(mol.TDA().__class__, tdscf.rhf.TDA)
        self.assertEqual(mol.dTDA().__class__, tdscf.rks.dTDA)
        self.assertEqual(mol.TDBP86().__class__, tdscf.rks.TDDFTNoHybrid)
        self.assertEqual(mol.TDB3LYP().__class__, tdscf.rks.TDDFT)
        self.assertRaises(AttributeError, lambda: mol.xyz)
        self.assertRaises(AttributeError, lambda: mol.TDxyz)

    def test_ao2mo(self):
        mol = gto.M(atom='He')
        nao = mol.nao
        eri = mol.ao2mo(numpy.eye(nao))
        self.assertAlmostEqual(eri[0,0], 1.0557129427350722, 12)

    def test_tofile(self):
        tmpfile = tempfile.NamedTemporaryFile()
        mol = gto.M(atom=[[1  , (0.,1.,1.)],
                          ["O1", (0.,0.,0.)],
                          [1  , (1.,1.,0.)], ])
        out1 = mol.tofile(tmpfile.name, format='xyz')
        ref = '''3
XYZ from PySCF
H           0.00000000        1.00000000        1.00000000
O           0.00000000        0.00000000        0.00000000
H           1.00000000        1.00000000        0.00000000
'''
        with open(tmpfile.name, 'r') as f:
            self.assertEqual(f.read(), ref)
        self.assertEqual(out1, ref[:-1])

        tmpfile = tempfile.NamedTemporaryFile(suffix='.zmat')
        str1 = mol.tofile(tmpfile.name, format='zmat')
        #FIXME:self.assertEqual(mol._atom, mol.fromfile(tmpfile.name))

    def test_frac_particles(self):
        mol = gto.M(atom=[['h', (0.,1.,1.)],
                          ['O', (0.,0.,0.)],
                          ['h', (1.,1.,0.)],],
                     basis='sto3g')
        mol._atm[1, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
        mol._env[mol._atm[1, gto.PTR_FRAC_CHARGE]] = 2.5
        self.assertAlmostEqual(mol.atom_charges().sum(), 4.5, 12)
        self.assertAlmostEqual(mol.atom_charge(1), 2.5, 12)

        # Add test after updating cint
        ref = 0
        for ia in range(mol.natm):
            with mol.with_rinv_origin(mol.atom_coord(ia)):
                ref -= mol.intor('int1e_rinv') * mol.atom_charge(ia)
        v = mol.intor('int1e_nuc')
        self.assertAlmostEqual(abs(ref-v).max(), 0, 12)

    def test_fromstring(self):
        mol = gto.Mole()
        mol.fromstring('2\n\nH 0 0 1\nH 0 -1 0')
        print(mol._atom == [('H', [0.0, 0.0, 1.8897261245650618]), ('H', [0.0, -1.8897261245650618, 0.0])])
        print(mol.atom == [('H', [0.0, 0.0, 1.0]), ('H', [0.0, -1.0, 0.0])])
        print(mol.unit == 'Angstrom')

    def test_fromfile(self):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xyz') as f:
            f.write('2\n\nH 0 0 1; H 0 -1 0')
            f.flush()
            mol = gto.Mole()
            mol.fromfile(f.name)
            print(mol._atom == [('H', [0.0, 0.0, 1.8897261245650618]), ('H', [0.0, -1.8897261245650618, 0.0])])
            print(mol.atom == [('H', [0.0, 0.0, 1.0]), ('H', [0.0, -1.0, 0.0])])
            print(mol.unit == 'Angstrom')

    def test_uncontract(self):
        basis = gto.basis.parse('''
H    S
0.9  0.8  0
0.5  0.5  0.6
0.3  0.5  0.8
H    S
0.3  1
H    P
0.9  0.6
0.5  0.6
0.3  0.6
''')
        self.assertEqual(gto.uncontract(basis),
                         [[0, [0.9, 1]], [0, [0.5, 1]], [0, [0.3, 1]],
                          [1, [0.9, 1]], [1, [0.5, 1]], [1, [0.3, 1]]])

        basis = [[1, 0, [0.9, .7], [0.5, .7]], [1, [0.5, .8], [0.3, .6]], [1, [0.3, 1]]]
        self.assertEqual(gto.uncontract(basis),
                         [[1, [0.9, 1]], [1, [0.5, 1]], [1, [0.3, 1]]])

        basis = [[1, -2, [0.9, .7], [0.5, .7]], [1, [0.5, .8], [0.3, .6]], [1, [0.3, 1]]]
        self.assertEqual(gto.uncontract(basis),
                         [[1, -2, [0.9, 1]], [1, -2, [0.5, 1]], [1, [0.3, 1]]])

        # FIXME:
        #basis = [[1, [0.9, .7], [0.5, .7]], [1, -2, [0.5, .8], [0.3, .6]], [1, [0.3, 1]]]
        #serl.assertEqual(gto.uncontract(basis),
        #                 [[1, [0.9, 1]], [1, [0.5, 1]], [1, [0.3, 1]]])

    def test_decontract_basis(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 01', basis='ccpvdz')
        pmol, ctr_coeff = mol.decontract_basis(atoms=[1], to_cart=True)
        ctr_coeff = scipy.linalg.block_diag(*ctr_coeff)
        s = ctr_coeff.T.dot(pmol.intor('int1e_ovlp')).dot(ctr_coeff)
        self.assertAlmostEqual(abs(s - mol.intor('int1e_ovlp')).max(), 0, 12)

        # discard basis on atom 2. (related to issue #1711)
        mol._bas = mol._bas[:5]
        pmol, c = mol.decontract_basis()
        self.assertEqual(pmol.nbas, 14)
        self.assertEqual(len(c), 5)

        mol = gto.M(atom='He',
                    basis=('ccpvdz', [[0, [5, 1]], [1, [3, 1]]]))
        pmol, contr_coeff = mol.decontract_basis()
        self.assertEqual(len(contr_coeff), 5)
        contr_coeff = scipy.linalg.block_diag(*contr_coeff)
        s = contr_coeff.T.dot(pmol.intor('int1e_ovlp')).dot(contr_coeff)
        self.assertAlmostEqual(abs(s - mol.intor('int1e_ovlp')).max(), 0, 12)

        mol = gto.M(atom='H 0 0 0; F 0 0 1', basis=[[0, (2, .5), (1, .5)],
                                                    [0, (2, .1), (1, .9)],
                                                    [0, (4., 1)]])
        with self.assertRaises(RuntimeError):
            mol.decontract_basis(aggregate=False)
        pmol, c = mol.decontract_basis(aggregate=True)
        self.assertEqual(pmol.nbas, 6)
        self.assertEqual(c.shape, (6, 6))
        s = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
        self.assertAlmostEqual(abs(s - mol.intor('int1e_ovlp')).max(), 0, 12)

    def test_ao_rotation_matrix(self):
        mol = gto.M(atom='O 0 0 0.2; H1 0 -.8 -.5; H2 0 .8 -.5', basis='ccpvdz')
        numpy.random.seed(1)
        axes = numpy.linalg.svd(numpy.random.rand(3,3))[0]
        mol1 = gto.M(atom=list(zip(['O', 'H', 'H'], mol.atom_coords().dot(axes.T))),
                     basis='ccpvdz', unit='Bohr')
        u = mol.ao_rotation_matrix(axes)
        v0 = u.T.dot(mol.intor('int1e_nuc')).dot(u)
        v1 = mol1.intor('int1e_nuc')
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 12)

if __name__ == "__main__":
    print("test mole.py")
    unittest.main()
