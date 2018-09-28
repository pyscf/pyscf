import unittest
import numpy as np

from pyscf.lib import finger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbc_df

import pyscf.cc
import pyscf.pbc.cc as pbcc
import make_test_cell
from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.cc.kccsd as kccsd
import pyscf.pbc.cc.eom_kccsd_ghf as kccsd_ghf


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
#cell.verbose = 7
#cell.output = '/dev/null'
cell.build()
nmp = [1,1,2]
thresh = 1e-8

def get_idx_r2(nkpts,nocc,nvir,ki,kj,i,j,a):
    o1 = nvir
    o2 = nocc*o1
    o3 = nocc*o2
    o4 = nkpts*o3
    return ki*o4 + ki*o3 + i*o2 + i*o1 + a

def get_ip_identity(nocc,nvir,nkpts,I):
    count = 0
    indices = []
    for i in range(nocc):
        indices.append(i)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        r1 = np.zeros(nocc,dtype=complex)
                        r2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)
                        if j >= i:
                            pass
                        else:
                            r2[ki,kj,i,j,a] = 1.0
                            r2[kj,ki,j,i,a] = -1.0
                            I[:,nocc + count] = kccsd_ghf.amplitudes_to_vector_ip(r1,r2)
                            indices.append(nocc + count)
                        count = count + 1
    return indices

class TestHe(unittest.TestCase):
    def _test_ip_diag(self,kmf):
        cc = kccsd.KGCCSD(kmf)
        Ecc = cc.kernel()[0]

        eom = kccsd_ghf.EOMIP(cc)
        imds = eom.make_imds()
        nkpts, nocc, nvir = imds.t1.shape
        diag = kccsd_ghf.ipccsd_diag(eom,0,imds=imds)
        
        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nocc,:nocc] = np.identity(nocc,dtype=complex)
        indices = get_ip_identity(nocc,nvir,nkpts,I)
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_ghf.ipccsd_matvec(eom,I[:,idx],0,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < thresh,"Difference in IP diag: {}".format(diff))

    def test_he_112_ip_diag(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_212_ip_diag(self):
        kpts = cell.make_kpts([2,1,2])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_131_ip_diag(self):
        kpts = cell.make_kpts([1,3,1])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_supercell_vs_kpt(self):
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        kmf = pbcscf.KGHF(cell, kpts=cell.make_kpts(nmp), exxdiv=None)
        kmf.kernel()
        mycc = pbcc.KGCCSD(kmf)
        mycc.conv_tol = 1e-12
        mycc.conv_tol_normt = 1e-10
        ecc2, t1, t2 = mycc.kernel()
        ecc_ref = -0.01044680113334205
        print ecc2
        self.assertAlmostEqual(abs(ecc_ref/2. - ecc2), 0, 10)


if __name__ == '__main__':
    unittest.main()
