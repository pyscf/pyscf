import unittest
import numpy as np

from pyscf.lib import finger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbc_df

import pyscf.cc
import pyscf.pbc.cc as pbcc
import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.cc.kccsd_uhf as kccsd
import pyscf.pbc.cc.eom_kccsd_uhf as kccsd_uhf


def setUpModule():
    global cell, KGCCSD_TEST_THRESHOLD
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.build()
    KGCCSD_TEST_THRESHOLD = 1e-8

def tearDownModule():
    global cell, KGCCSD_TEST_THRESHOLD
    del cell, KGCCSD_TEST_THRESHOLD

class KnownValues(unittest.TestCase):
    def _test_ip_diag(self,kmf,kshift=0):
        cc = kccsd.KUCCSD(kmf)
        Ecc = cc.kernel()[0]

        eom = kccsd_uhf.EOMIP(cc)
        imds = eom.make_imds()
        t1a,t1b = imds.t1
        nkpts, nocc_a, nvir_a = t1a.shape
        nkpts, nocc_b, nvir_b = t1b.shape
        nocc = nocc_a + nocc_b
        diag = kccsd_uhf.ipccsd_diag(eom,kshift,imds=imds)

        I = np.identity(diag.shape[0],dtype=complex)
        indices = np.arange(diag.shape[0])
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_uhf.ipccsd_matvec(eom,I[:,idx],kshift,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < KGCCSD_TEST_THRESHOLD,"Difference in IP diag: {}".format(diff))

    def _test_ea_diag(self,kmf,kshift=0):
        cc = kccsd.KUCCSD(kmf)
        Ecc = cc.kernel()[0]

        eom = kccsd_uhf.EOMEA(cc)
        imds = eom.make_imds()
        t1a,t1b = imds.t1
        nkpts, nocc_a, nvir_a = t1a.shape
        nkpts, nocc_b, nvir_b = t1b.shape
        nocc = nocc_a + nocc_b
        nvir = nvir_a + nvir_b
        diag = kccsd_uhf.eaccsd_diag(eom,kshift,imds=imds)

        I = np.identity(diag.shape[0],dtype=complex)
        indices = np.arange(diag.shape[0])
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_uhf.eaccsd_matvec(eom,I[:,idx],kshift,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < KGCCSD_TEST_THRESHOLD,"Difference in EA diag: {}".format(diff))

    def test_he_112_ip_diag(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_112_ip_diag_shift(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf,kshift=1)

    def test_he_212_ip_diag_high_cost(self):
        kpts = cell.make_kpts([2,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_131_ip_diag(self):
        kpts = cell.make_kpts([1,3,1])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_112_ea_diag(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ea_diag(kmf)

    def test_he_112_ea_diag_shift(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ea_diag(kmf,kshift=1)

    def test_he_212_ea_diag_high_cost(self):
        kpts = cell.make_kpts([2,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ea_diag(kmf)

    def test_he_131_ea_diag(self):
        kpts = cell.make_kpts([1,3,1])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ea_diag(kmf)

if __name__ == '__main__':
    unittest.main()
