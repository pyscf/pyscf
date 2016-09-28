#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import lib
import pyscf.lib.parameters as param

mol0 = gto.Mole()
mol0.atom = [
    [1  , (0.,1.,1.)],
    ["O1", (0.,0.,0.)],
    [1  , (1.,1.,0.)], ]
mol0.nucmod = { "O":'gaussian', 3:'g' }
mol0.unit = 'ang'
mol0.basis = {
    "O": [(0, 0, (15, 1)), ],
    "H": [(0, 0, (1, 1, 0), (3, 3, 1), (5, 1, 0)),
          (1, 0, (1, 1)), ]}
mol0.symmetry = 1
mol0.charge = 1
mol0.spin = 1
mol0.basis['O'].extend(gto.mole.expand_etbs(((0, 4, 1, 1.8),
                                            (1, 3, 2, 1.8),
                                            (2, 2, 1, 1.8),)))
mol0.verbose = 4
mol0.ecp = {'O1': 'lanl2dz'}
mol0.output = None
mol0.build()

class KnowValues(unittest.TestCase):
    def test_intor_cross(self):
        mol1 = mol0.unpack(mol0.pack())
        mol1.atom = '''
                1    0  1  1
                O    0  0  0
                h    1  1  0'''
        mol1.basis = {'O': gto.basis.parse('''
C    S
   3047.5249000              0.0018347        
    457.3695100              0.0140373        
    103.9486900              0.0688426        
     29.2101550              0.2321844        
      9.2866630              0.4679413        
      3.1639270              0.3623120        
C    SP
      7.8682724             -0.1193324              0.0689991        
      1.8812885             -0.1608542              0.3164240        
      0.5442493              1.1434564              0.7443083        
C    SP
      0.1687144              1.0000000              1.0000000'''),
                      'H': '6-31g'}
        mol1.build()
        v = gto.mole.intor_cross('cint1e_ovlp_sph', mol0, mol1)
        self.assertAlmostEqual(numpy.linalg.norm(v), 3.6489423434168562, 1)

    def test_num_basis(self):
        self.assertEqual(mol0.nao_nr(), 34)
        self.assertEqual(mol0.nao_2c(), 68)

    def test_time_reversal_map(self):
        tao = [-2, 1, -4, 3, 6, -5, 10, -9, 8, -7, -12, 11, -14, 13, -16, 15, -18, 17,
               -20, 19, 22, -21, 26, -25, 24, -23, 28, -27, 32, -31, 30, -29, 34, -33,
               38, -37, 36, -35, -42, 41, -40, 39, -48, 47, -46, 45, -44, 43, -52, 51,
               -50, 49, -58, 57, -56, 55, -54, 53, -60, 59, -62, 61, 64, -63, 68, -67,
               66, -65]
        self.assertEqual(mol0.time_reversal_map(), tao)

    def test_check_sanity(self):
        mol1 = mol0.copy()
        mol1.x = None
        mol1.copy = None
        mol1.check_sanity()

    def test_nao_range(self):
        self.assertEqual(mol0.nao_nr_range(1,4), (2, 7))
        self.assertEqual(mol0.nao_2c_range(1,4), (4, 14))
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_nr()), 2151)
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_2c()), 4302)

    def test_search_bas(self):
        self.assertEqual(mol0.search_shell_id(1, 1), 7)
        self.assertEqual(mol0.search_ao_nr(1, 1, -1, 5), None)
        self.assertEqual(mol0.search_ao_nr(1, 1, -1, 4), 16)

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

    def test_given_symmetry(self):
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
          H -0.9444878100 0.0000000000 1.3265673200
          H -0.9444878100 0.0000000000 -1.3265673200''',charge=1)
        mol2 = gto.M(atom='''H 0.0000000000 0.0000000000 0.0000000000
          H 0.9444878100 1.3265673200 0.0000000000
          H 0.9497795800 -1.3265673200 0.0000000000
          H -0.9444878100 0.0000000000 1.3265673200
          H -0.9444878100 0.0000000000 -1.3265673200''',charge=1)
        self.assertTrue(gto.same_mol(mol1, mol2))

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

    def test_format_atom(self):
        atoms = [['h' , 0,1,1], ["O1", (0.,0.,0.)], [1, 1.,1.,0.],]
        self.assertTrue(numpy.allclose(gto.mole.format_atom(atoms, unit='Ang')[0][1],
                                       [0.0, 1.8897261245650618, 1.8897261245650618]))
        atoms = '''h 0 1 1
        O1 0 0 0; 1 1 1 0'''
        self.assertTrue(numpy.allclose(gto.mole.format_atom(atoms, unit=1)[0][1],
                                       [0.0, 1., 1.]))
        atoms = 'O1; h 1 1; 1 1 1 2 90'
        atoms = gto.mole.format_atom(atoms, unit=1)[2]
        self.assertEqual(atoms[0], 'H')
        self.assertTrue(numpy.allclose(atoms[1], [0, 0, 1.]))

    def test_default_basis(self):
        mol = gto.M(atom=[['h' , 0,1,1], ["O1", (0.,0.,0.)], [1, 1.,1.,0.],],
                    basis={'default':'321g', 'O1': 'sto3g'})
        self.assertEqual(sorted(mol._basis.keys()), ['H', 'O1'])


if __name__ == "__main__":
    print("test mole.py")
    unittest.main()
