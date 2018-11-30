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
import pyscf.pbc.cc.kccsd_uhf as kccsd
import pyscf.pbc.cc.eom_kccsd_uhf as kccsd_uhf


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
thresh = 1e-8

def get_idx_r2(nkpts,nocc,nvir,ki,kj,i,j,a):
    o1 = nvir
    o2 = nocc*o1
    o3 = nocc*o2
    o4 = nkpts*o3
    return ki*o4 + ki*o3 + i*o2 + i*o1 + a

def get_ip_identity(nocc_a,nocc_b,nvir_a,nvir_b,nkpts,I):
    count = 0
    indices = []
    offset = nocc_a

    # a
    for i in range(nocc_a):
        indices.append(i)
    # b
    for i in range(nocc_b):
        indices.append(i + offset)
    nocc = nocc_a + nocc_b
    offset = nocc
    for ki in range(nkpts):
        for kj in range(nkpts):
            # aaa
            for i in range(nocc_a):
                for j in range(nocc_a):
                    for a in range(nvir_a):
                        r1a = np.zeros(nocc_a,dtype=complex)
                        r1b = np.zeros(nocc_a,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a),dtype=complex)
                        r2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a),dtype=complex)
                        r2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b),dtype=complex)
                        if j >= i:
                            pass
                        else:
                            r2aaa[ki,kj,i,j,a] = 1.0
                            r2aaa[kj,ki,j,i,a] = -1.0
                            I[:,nocc + count] = kccsd_uhf.amplitudes_to_vector_ip(
                                    (r1a,r1b),(r2aaa,r2baa,r2abb,r2bbb))
                            indices.append(offset + count)
                        count = count + 1
    for ki in range(nkpts):
        for kj in range(nkpts):
            # baa
            for i in range(nocc_b):
                for j in range(nocc_a):
                    for a in range(nvir_a):
                        r1a = np.zeros(nocc_a,dtype=complex)
                        r1b = np.zeros(nocc_a,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a),dtype=complex)
                        r2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b),dtype=complex)
                        r2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b),dtype=complex)
                        #if j >= i:
                        #    pass
                        #else:
                        r2baa[ki,kj,i,j,a] = 1.0
                        #r2baa[kj,ki,j,i,a] = -1.0
                        I[:,nocc + count] = kccsd_uhf.amplitudes_to_vector_ip(
                                (r1a,r1b),(r2aaa,r2baa,r2abb,r2bbb))
                        indices.append(offset + count)
                        count = count + 1
    for ki in range(nkpts):
        for kj in range(nkpts):
            # abb
            for i in range(nocc_a):
                for j in range(nocc_b):
                    for a in range(nvir_b):
                        r1a = np.zeros(nocc_a,dtype=complex)
                        r1b = np.zeros(nocc_a,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a),dtype=complex)
                        r2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b),dtype=complex)
                        r2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b),dtype=complex)
                        #if j >= i:
                        #    pass
                        #else:
                        r2abb[ki,kj,i,j,a] = 1.0
                        #r2abb[kj,ki,j,i,a] = -1.0
                        I[:,nocc + count] = kccsd_uhf.amplitudes_to_vector_ip(
                                (r1a,r1b),(r2aaa,r2baa,r2abb,r2bbb))
                        indices.append(offset + count)
                        count = count + 1
    for ki in range(nkpts):
        for kj in range(nkpts):
            # bbb
            for i in range(nocc_b):
                for j in range(nocc_b):
                    for a in range(nvir_b):
                        r1a = np.zeros(nocc_a,dtype=complex)
                        r1b = np.zeros(nocc_a,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a),dtype=complex)
                        r2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b),dtype=complex)
                        r2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b),dtype=complex)
                        if j >= i:
                            pass
                        else:
                            r2bbb[ki,kj,i,j,a] = 1.0
                            r2bbb[kj,ki,j,i,a] = -1.0
                            I[:,nocc + count] = kccsd_uhf.amplitudes_to_vector_ip(
                                    (r1a,r1b),(r2aaa,r2baa,r2abb,r2bbb))
                            indices.append(offset + count)
                        count = count + 1
    return indices

class TestHe(unittest.TestCase):
    def _test_ip_diag(self,kmf):
        cc = kccsd.KUCCSD(kmf)
        Ecc = cc.kernel()[0]

        eom = kccsd_uhf.EOMIP(cc)
        imds = eom.make_imds()
        t1a,t1b = imds.t1
        nkpts, nocc_a, nvir_a = t1a.shape
        nkpts, nocc_b, nvir_b = t1b.shape
        nocc = nocc_a + nocc_b
        diag = kccsd_uhf.ipccsd_diag(eom,0,imds=imds)
        
        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nocc,:nocc] = np.identity(nocc,dtype=complex)
        indices = get_ip_identity(nocc_a,nocc_b,nvir_a,nvir_b,nkpts,I)
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_uhf.ipccsd_matvec(eom,I[:,idx],0,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        #print(diag_ref)
        #print(diag_out)
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < thresh,"Difference in IP diag: {}".format(diff))

    def test_he_112_ip_diag(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_212_ip_diag(self):
        kpts = cell.make_kpts([2,1,2])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

    def test_he_131_ip_diag(self):
        kpts = cell.make_kpts([1,3,1])
        kmf = pbcscf.KUHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        self._test_ip_diag(kmf)

if __name__ == '__main__':
    unittest.main()
