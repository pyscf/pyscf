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
cell.build()
thresh = 1e-8

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
                        r2baa[ki,kj,i,j,a] = 1.0
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
                        r2abb[ki,kj,i,j,a] = 1.0
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

def get_ea_identity(kshift,nocc_a,nocc_b,nvir_a,nvir_b,nkpts,I,cc):
    count = 0
    indices = []
    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    # a
    for i in range(nvir_a):
        indices.append(i)
    # b
    offset = nvir_a
    for i in range(nvir_b):
        indices.append(i + offset)
    offset = nvir
    for kj in range(nkpts):
        for ka in range(nkpts):
            # aaa
            for j in range(nocc_a):
                for a in range(nvir_a):
                    for b in range(nvir_a):
                        r1a = np.zeros(nvir_a,dtype=complex)
                        r1b = np.zeros(nvir_b,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nvir_a,nvir_a),dtype=complex)
                        r2aba = np.zeros((nkpts,nkpts,nocc_a,nvir_b,nvir_a),dtype=complex)
                        r2bab = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nvir_b,nvir_b),dtype=complex)
                        if b >= a:
                            pass
                        else:
                            kb = kconserv[kshift,ka,kj]
                            r2aaa[kj,ka,j,a,b] = 1.0
                            r2aaa[kj,kb,j,b,a] = -1.0
                            I[:,nvir + count] = kccsd_uhf.amplitudes_to_vector_ea(
                                    (r1a,r1b),(r2aaa,r2aba,r2bab,r2bbb))
                            indices.append(offset + count)
                        count = count + 1
    for kj in range(nkpts):
        for ka in range(nkpts):
            # aba
            for j in range(nocc_a):
                for a in range(nvir_b):
                    for b in range(nvir_a):
                        r1a = np.zeros(nvir_a,dtype=complex)
                        r1b = np.zeros(nvir_b,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nvir_a,nvir_a),dtype=complex)
                        r2aba = np.zeros((nkpts,nkpts,nocc_a,nvir_b,nvir_a),dtype=complex)
                        r2bab = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2aba[kj,ka,j,a,b] = 1.0
                        I[:,nvir + count] = kccsd_uhf.amplitudes_to_vector_ea(
                                (r1a,r1b),(r2aaa,r2aba,r2bab,r2bbb))
                        indices.append(offset + count)
                        count = count + 1
    for kj in range(nkpts):
        for ka in range(nkpts):
            # bab
            for j in range(nocc_b):
                for a in range(nvir_a):
                    for b in range(nvir_b):
                        r1a = np.zeros(nvir_a,dtype=complex)
                        r1b = np.zeros(nvir_b,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nvir_a,nvir_a),dtype=complex)
                        r2aba = np.zeros((nkpts,nkpts,nocc_a,nvir_b,nvir_a),dtype=complex)
                        r2bab = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2bab[kj,ka,j,a,b] = 1.0
                        I[:,nvir + count] = kccsd_uhf.amplitudes_to_vector_ea(
                                (r1a,r1b),(r2aaa,r2aba,r2bab,r2bbb))
                        indices.append(offset + count)
                        count = count + 1
    for kj in range(nkpts):
        for ka in range(nkpts):
            # bbb
            for j in range(nocc_b):
                for a in range(nvir_b):
                    for b in range(nvir_b):
                        r1a = np.zeros(nvir_a,dtype=complex)
                        r1b = np.zeros(nvir_b,dtype=complex)
                        r2aaa = np.zeros((nkpts,nkpts,nocc_a,nvir_a,nvir_a),dtype=complex)
                        r2aba = np.zeros((nkpts,nkpts,nocc_a,nvir_b,nvir_a),dtype=complex)
                        r2bab = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b),dtype=complex)
                        r2bbb = np.zeros((nkpts,nkpts,nocc_b,nvir_b,nvir_b),dtype=complex)
                        if b >= a:
                            pass
                        else:
                            kb = kconserv[kshift,ka,kj]
                            r2bbb[kj,ka,j,a,b] = 1.0
                            r2bbb[kj,kb,j,b,a] = -1.0
                            I[:,nvir + count] = kccsd_uhf.amplitudes_to_vector_ea(
                                    (r1a,r1b),(r2aaa,r2aba,r2bab,r2bbb))
                            indices.append(offset + count)
                        count = count + 1
    return indices

class TestHe(unittest.TestCase):
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
        
        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nocc,:nocc] = np.identity(nocc,dtype=complex)
        indices = get_ip_identity(nocc_a,nocc_b,nvir_a,nvir_b,nkpts,I)
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_uhf.ipccsd_matvec(eom,I[:,idx],kshift,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < thresh,"Difference in IP diag: {}".format(diff))

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
 
        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nvir,:nvir] = np.identity(nvir,dtype=complex)
        indices = get_ea_identity(kshift,nocc_a,nocc_b,nvir_a,nvir_b,nkpts,I,cc)
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_uhf.eaccsd_matvec(eom,I[:,idx],kshift,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < thresh,"Difference in EA diag: {}".format(diff))

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
