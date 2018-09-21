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
cell.output = '/dev/null'
cell.build()
thresh = 1e-8

# Helper functions
#def kconserve_pmatrix(nkpts, kconserv):
#    Ps = np.zeros((nkpts, nkpts, nkpts, nkpts))
#    for ki in range(nkpts):
#        for kj in range(nkpts):
#            for ka in range(nkpts):
#                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
#                kb = kconserv[ki, ka, kj]
#                Ps[ki, kj, ka, kb] = 1
#    return Ps

def get_idx_r2(nkpts,nocc,nvir,ki,kj,i,j,a):
    o1 = nvir
    o2 = nocc*o1
    o3 = nocc*o2
    o4 = nkpts*o3
    return ki*o4 + ki*o3 + i*o2 + i*o1 + a

class TestHe(unittest.TestCase):
    def test_he_112(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()

        cc = kccsd.KGCCSD(kmf)
        Ecc = cc.kernel()[0]
        #print(Escf,Ecc)

        eom = kccsd_ghf.EOMIP(cc)
        imds = eom.make_imds()
        nkpts, nocc, nvir = imds.t1.shape
        diag = kccsd_ghf.ipccsd_diag(eom,0,imds=imds)

        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nocc,:nocc] = np.identity(nocc,dtype=complex)
        count = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                for i in range(nocc):
                    for j in range(nocc):
                        for a in range(nvir):
                            r1 = np.zeros(nocc)
                            r2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir))
                            if ki == kj and i == j:
                                pass
                            else:
                                r2[ki,kj,i,j,a] = 1.0
                                r2[kj,ki,j,i,a] = -1.0
                            I[:,nocc + count] = kccsd_ghf.amplitudes_to_vector_ip(r1,r2)
                            count = count + 1
                            #ip = ki*o4 + kj*o3 + i*o2 + j*o1 + a
                            #ip2 = kj*o4 + ki*o3 + j*o2 + i*o1 + a
                            #if ip == ip2:
                            #    continue
                            #I[nocc + ip,nocc + ip] = 1.0
                            #I[nocc + ip2,nocc + ip] = -1.0
        H = np.zeros(I.shape,dtype=complex)
        for i in range(diag.shape[0]):
            H[:,i] = kccsd_ghf.ipccsd_matvec(eom,I[:,i],0,imds=imds)

        diag_ref = H.diagonal().copy()
        for ki in range(nkpts):
            for i in range(nocc):
                for a in range(nvir):
                    idx = get_idx_r2(nkpts,nocc,nvir,ki,kj,i,j,a)
                    diag[nocc + idx] = 0.0
        diff = np.linalg.norm(diag_ref - diag)
        print(diag_ref - diag)

        self.assertTrue(abs(diff) < thresh,"Difference in IP diag: {}".format(diff))

if __name__ == '__main__':
    unittest.main()
