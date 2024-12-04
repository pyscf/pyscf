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

if __name__ == "__main__":
    print("Full Tests for qcschema")
    unittest.main()
