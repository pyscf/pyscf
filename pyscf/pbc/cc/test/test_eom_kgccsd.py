import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc.eom_kccsd_ghf import EOMIP, EOMEA, EOMEE
from pyscf.pbc.cc.eom_kccsd_ghf import EOMIP_Ta, EOMEA_Ta
from pyscf.cc import eom_gccsd
import unittest

def setUpModule():
    global cell, kmf, mycc, eris
    cell = make_test_cell.test_cell_n3_diffuse()
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2], with_gamma_point=True), exxdiv=None)
    kmf.conv_tol = 1e-10
    kmf.conv_tol_grad = 1e-6
    kmf.scf()

    mycc = cc.KGCCSD(kmf)
    mycc.conv_tol = 1e-7
    mycc.conv_tol_normt = 1e-7
    mycc.run()
    eris = mycc.ao2mo()
    eris.mo_energy = [eris.fock[ikpt].diagonal().real for ikpt in range(mycc.nkpts)]

def tearDownModule():
    global cell, kmf, mycc, eris
    cell.stdout.close()
    del cell, kmf, mycc, eris

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        ecc2 = mycc.e_corr
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP(mycc)
        imds = eom.make_imds(eris=eris)
        # Basic ipccsd
        e1_obt, v = eom.ipccsd(nroots=3, left=True, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0],-1.14894700482871,6)
        self.assertAlmostEqual(e1_obt[0][1],-1.148947004822481,6)
        self.assertAlmostEqual(e1_obt[0][2],-1.108819439453179,6)

        # Ensure left is working
        e1_obt, v = eom.ipccsd(nroots=3, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -1.1489469962099519, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1489469961858796, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.1088194518036925, 6)

        # Ensure kptlist behaves correctly
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -0.9074337292436309, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.9074337292161299, 6)
        self.assertAlmostEqual(e1_obt[0][2], -0.9074331788469051, 6)

        eom = EOMEA(mycc)
        imds = eom.make_imds(eris=eris)
        # Basic eaccsd
        e2_obt, v = eom.eaccsd(nroots=3, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.2669788613362731, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2669788614703625, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.278883205515518, 6)

        # Ensure left is working
        e2_obt, v = eom.eaccsd(nroots=3, left=True, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.266978976813125,6)
        self.assertAlmostEqual(e2_obt[0][1], 1.266978976822988,6)
        self.assertAlmostEqual(e2_obt[0][2], 1.278883205348326,6)

        # Ensure kptlist behaves correctly
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.227583017804503, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2275830178298166, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.3830379190440196, 6)

        # Basis eeccsd
        eom = EOMEE(mycc)
        imds = eom.make_imds(eris=eris)
        ee, v = eom.eeccsd(nroots=3, kptlist=[0], imds=imds)
        self.assertAlmostEqual(ee[0][0], 0.118301677904104, 6)
        self.assertAlmostEqual(ee[0][1], 0.118301914631351, 6)
        self.assertAlmostEqual(ee[0][2], 0.128285117266903, 6)
        ee, v = eom.eeccsd(nroots=3, kptlist=[1], imds=imds)
        self.assertAlmostEqual(ee[0][0], 0.07928010716890202, 6)
        self.assertAlmostEqual(ee[0][1], 0.07928011416043479, 6)
        self.assertAlmostEqual(ee[0][2], 0.07928011417159982, 6)

    def test_n3_diffuse_frozen(self):
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc_frozen = cc.KGCCSD(kmf, frozen=[[0,1],[0,1,2,3]])
        mycc_frozen.conv_tol = 1e-7
        mycc_frozen.conv_tol_normt = 1e-7
        eris = mycc_frozen.ao2mo()
        eris.mo_energy = [eris.fock[ikpt].diagonal().real for ikpt in range(mycc_frozen.nkpts)]
        ecc2, t1, t2 = mycc_frozen.kernel(eris=eris)
        self.assertAlmostEqual(ecc2, -0.0442506265840587, 6)

        eom = EOMIP(mycc_frozen)
        imds = eom.make_imds(eris=eris)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=False, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.104163717600433, 6)

        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -0.8983145129187627, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.8983145129187627, 6)
        self.assertAlmostEqual(e1_obt[0][2], -0.8983139520017552, 6)

        eom = EOMEA(mycc_frozen)
        imds = eom.make_imds(eris=eris)
        e2_obt, v = eom.eaccsd(nroots=3, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.2572812499753756, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2572812532456588, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.280747357928012, 6)

        eom = EOMEA(mycc_frozen)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.229802629928757, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.229802629928764, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.384394578043613, 6)

    def test_n3_diffuse_star(self):
        '''Tests EOM-CCSD* method.'''
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        ecc2 = mycc.e_corr
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP(mycc)
        e1_obt = eom.ipccsd_star(nroots=3, koopmans=True, kptlist=[0], eris=eris)
        self.assertAlmostEqual(e1_obt[0][0], -1.1452481194582802, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1452481194456137, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.1174912094746994, 6)

        eom = EOMEA(mycc)
        e1_obt = eom.eaccsd_star(nroots=2, koopmans=True, kptlist=[0,1], eris=eris)
        self.assertAlmostEqual(e1_obt[0][0], 1.260627794895514, 6)
        self.assertAlmostEqual(e1_obt[0][1], 1.260627794895514, 6)
        self.assertAlmostEqual(e1_obt[1][0], 1.2222607619733454, 6)
        self.assertAlmostEqual(e1_obt[1][1], 1.2222607619733026, 6)

    def test_n3_diffuse_Ta(self):
        '''Tests EOM-CCSD(T)*a method.'''
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        ecc2 = mycc.e_corr
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP_Ta(mycc)
        imds = eom.make_imds(eris=eris)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -1.146351234409813, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.146351234404151, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.107255699646373, 6)

        e1_obt = eom.ipccsd_star(nroots=3, koopmans=True, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e1_obt[0][0], -1.143510075691, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.143510075684, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.116991306080, 6)

        eom = EOMEA_Ta(mycc)
        imds = eom.make_imds(eris=eris)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[0], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.267728934041309, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.267728934041309, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.280954980102639, 6)

        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.2290479727093149, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2290479727093468, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.384154366703175, 6)

        e2_obt = eom.eaccsd_star(nroots=3, koopmans=True, kptlist=[1], imds=imds)
        self.assertAlmostEqual(e2_obt[0][0], 1.2229050426609025, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2229050426609025, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.374851059956632, 6)

if __name__ == '__main__':
    print("eom_kccsd_rhf tests")
    unittest.main()
