'''
    Loads QCSchema format json result and computes dipole moment.
    Wavefunction info is in QCSchema json data file.
'''
from os.path import join, abspath
from pyscf.tools.qcschema import *
from pyscf import gto, dft, lib
import numpy as np
import unittest

class KnownValues(unittest.TestCase):
    def test_qcschema_dipole(self):
        chkfile = ""
        qcschema_json = abspath(join(__file__, "..", "qcschema_result.json"))

        # Load Accelerated DFT output json
        qcschema_dict = load_qcschema_json(qcschema_json)

        # Create DFT object
        mol = recreate_mol_obj(qcschema_dict)
        ks = recreate_scf_obj(qcschema_dict,mol)

        #### Compute Molecular Dipole Moment ####
        # First compute density matrix
        mo_occ = ks.mo_occ
        mo_coeff = ks.mo_coeff
        dm = ks.make_rdm1(mo_coeff, mo_occ)
        DipMom = ks.dip_moment(ks.mol, dm, unit='Debye')
        #############

        known_dipole = [0.00000, -0.00000, -1.08948]
        # check each value individually
        for i in range(len(known_dipole)):
            self.assertAlmostEqual(known_dipole[i],DipMom[i],delta=1e-4)


    def test_trajectory_selection(self):
        def geometry(distance):
            return {'symbols': ['H', 'H'], 'geometry': [0, 0, 0, 0, 0, distance]}
        schema = {'molecule': geometry(1.4), 'initial_molecule': geometry(2.),
                  'final_molecule': geometry(1.5),
                  'trajectory': [{'molecule': geometry(1.8)}, {'molecule': geometry(1.6)}]}
        for select, distance in ((1, 1.4), (2, 2.), (3, 1.5), (4, 1.6)):
            with self.subTest(selection=select):
                atoms = load_qcschema_molecule(schema, mol_select=select, step=1)
                mol = gto.M(atom=atoms, unit='Bohr', basis='sto-3g', verbose=0)
                self.assertAlmostEqual(mol.atom_coords()[1, 2], distance, places=12)
                xyz = load_qcschema_molecule(schema, mol_select=select, step=1,
                                                     to_Angstrom=True, xyz=True)
                self.assertEqual(xyz.splitlines()[0], '2')
                restored = gto.M(atom='\n'.join(xyz.splitlines()[2:]), basis='sto-3g', verbose=0)
                np.testing.assert_allclose(restored.atom_coords(), mol.atom_coords(), atol=1e-12)

if __name__ == "__main__":
    print("Full Tests for qcschema")
    unittest.main()
