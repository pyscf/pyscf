## Loads QCSchema format json result and computes dipole moment.
## Wavefunction info is in QCSchema json data file.
from pyscf.lib.libqcschema import *
from pyscf import gto, dft, lib
import json
import numpy as np
import unittest
import tempfile

class KnownValues(unittest.TestCase):
    def test_libqcschema_dipole(self):
        chkfile = ""
        qcschema_json = "qcschema_result.json"

        # Load Accelerated DFT output json
        qcschema_dict = load_qcschema_json(qcschema_json)

        # Create DFT object
        mol, ks = recreate_scf_obj(qcschema_dict)

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
    print("Full Tests for libqcschema")
    unittest.main()
