import numpy as np
import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf import lib
from pyscf import lo
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto
from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.eom_kccsd_uhf import EOMIP, EOMEA
from pyscf.pbc.lib import kpts_helper
from pyscf.cc import eom_uccsd
import unittest

def setUpModule():
    global cell_n3d, kmf
    cell_n3d = make_test_cell.test_cell_n3_diffuse()
    kmf = pbcscf.KRHF(cell_n3d, cell_n3d.make_kpts((1,1,2), with_gamma_point=True), exxdiv=None)
    kmf.conv_tol = 1e-10
    kmf.scf()

def tearDownModule():
    global cell_n3d, kmf
    cell_n3d.stdout.close()
    del cell_n3d, kmf

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):
        self.assertAlmostEqual(kmf.e_tot, -6.1870676561721227, 6)
        mycc = pbcc.KUCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0676483711263548, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.148946994550331, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.148946994544336, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.10881945217663, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.26697897719539, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.26697897720534, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.278883205460933, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e1_obt[0][0], -0.9074337295664157, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.9074337295664148, 6)
        self.assertAlmostEqual(e1_obt[0][2], -0.9074331791699104, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.227583017460536, 5)
        self.assertAlmostEqual(e2_obt[0][1], 1.227583017460617, 5)
        self.assertAlmostEqual(e2_obt[0][2], 1.383037918699404, 5)

    def test_n3_diffuse_frozen(self):
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = pbcc.KUCCSD(kmf, frozen=([[0,],[0,1]],[[0],[0,1]]))
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0442506265840587, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1041637212148683, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.104163717600433, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.2572812499753756, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.257281253091707, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.2807473549827182, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=2, kptlist=[1])
        self.assertAlmostEqual(e1_obt[0][0], -1.04309071165136, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.04309071165136, 6)
        #self.assertAlmostEqual(e1_obt[0][2], -0.09623848794144876, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.229802629928757, 5)
        self.assertAlmostEqual(e2_obt[0][1], 1.229802629928764, 5)
        self.assertAlmostEqual(e2_obt[0][2], 1.384394578043613, 5)

    def test_eomea_matvec(self):
        cell = gto.Cell()
        cell.atom='''
        He 0.000000000000   0.000000000000   0.000000000000
        He 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.mesh = [13]*3
        cell.precision = 1e-10
        cell.build()

        np.random.seed(2)
        kmf = pbcscf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
        nmo = cell.nao_nr()
        kmf.mo_occ = np.zeros((2,3,nmo))
        kmf.mo_occ[0,:,:3] = 1
        kmf.mo_occ[1,:,:1] = 1
        kmf.mo_energy = np.arange(nmo) + np.random.random((2,3,nmo)) * .3
        kmf.mo_energy[kmf.mo_occ == 0] += 2

        mo = (np.random.random((2,3,nmo,nmo)) +
              np.random.random((2,3,nmo,nmo))*1j - .5-.5j)
        s = kmf.get_ovlp()
        kmf.mo_coeff = np.empty_like(mo)
        nkpts = len(kmf.kpts)
        for k in range(nkpts):
            kmf.mo_coeff[0,k] = lo.orth.vec_lowdin(mo[0,k], s[k])
            kmf.mo_coeff[1,k] = lo.orth.vec_lowdin(mo[1,k], s[k])

        def rand_t1_t2(mycc):
            nkpts = mycc.nkpts
            nocca, noccb = mycc.nocc
            nmoa, nmob = mycc.nmo
            nvira, nvirb = nmoa - nocca, nmob - noccb
            np.random.seed(1)
            t1a = (np.random.random((nkpts,nocca,nvira)) +
                   np.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
            t1b = (np.random.random((nkpts,noccb,nvirb)) +
                   np.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
            t2aa = (np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
                    np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
            kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
            t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
            tmp = t2aa.copy()
            for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[ki, kk, kj]
                t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
            t2ab = (np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
                    np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
            t2bb = (np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
                    np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
            t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
            tmp = t2bb.copy()
            for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[ki, kk, kj]
                t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

            t1 = (t1a, t1b)
            t2 = (t2aa, t2ab, t2bb)
            return t1, t2


        mycc = pbcc.KUCCSD(kmf)
        t1, t2 = rand_t1_t2(mycc)
        mycc.t1 = t1
        mycc.t2 = t2

        eris = mycc.ao2mo()
        eom = EOMEA(mycc)
        imds = eom.make_imds(eris)
        np.random.seed(9)
        vector = np.random.random(eom.vector_size())

        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (4.126336947439054 +0.5931985341760211j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (1.248516714348047  +2.310336141756983j), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.4529892564020126-5.093287166283228j), 7)


        kmf = kmf.density_fit(auxbasis=[[0, (2., 1.)], [0, (1., 1.)], [0, (.5, 1.)]])
        mycc._scf = kmf
        mycc.max_memory = 0
        eris = mycc.ao2mo()
        imds = eom.make_imds(eris)
        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (4.045928342346641 +0.5861843966140339j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (1.2695743252320795+2.28060203958305j  ), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.435385905375094-5.0991524119952505j), 7)

        mycc.max_memory = 4000
        eris = mycc.ao2mo()
        imds = eom.make_imds(eris)
        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (4.045928342346641 +0.5861843966140339j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (1.2695743252320795+2.28060203958305j  ), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.435385905375094-5.0991524119952505j), 7)

if __name__ == '__main__':
    print("eom_kccsd_uhf tests")
    unittest.main()
