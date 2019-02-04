import make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc.eom_kccsd_ghf import EOMIP, EOMEA
from pyscf.pbc.cc.eom_kccsd_ghf import EOMIP_Ta, EOMEA_Ta
from pyscf.cc import eom_gccsd
import unittest

cell = make_test_cell.test_cell_n3_diffuse()
kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2], with_gamma_point=True), exxdiv=None)
kmf.scf()

def tearDownModule():
    global cell, kmf
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        '''
        # treating 1*1*2 supercell at gamma point
        supcell = super_cell(cell,nmp)
        gmf  = scf.GHF(supcell,exxdiv=None)
        ehf  = gmf.kernel()
        gcc  = cc.GCCSD(gmf)
        gcc.conv_tol=1e-12
        gcc.conv_tol_normt=1e-10
        gcc.max_cycle=250
        ecc, t1, t2 = gcc.kernel()
        print('GHF energy (supercell) %.7f \n' % (float(ehf)/2.))
        print('GCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))

        eom = eom_gccsd.EOMIP(gcc)
        e1, v = eom.ipccsd(nroots=6)
        eom = eom_gccsd.EOMEA(gcc)
        e2, v = eom.eaccsd(nroots=6, left=True, koopmans=True)
        '''
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = cc.KGCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP(mycc)
        imds = eom.make_imds()
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
        imds = eom.make_imds()
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

    def test_n3_diffuse_frozen(self):
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = cc.KGCCSD(kmf, frozen=[[0,1],[0,1,2,3]])
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0442506265840587, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=False, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.104163717600433, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.2572812499753756, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2572812532456588, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.280747357928012, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e1_obt[0][0], -0.8983145129187627, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.8983145129187627, 6)
        self.assertAlmostEqual(e1_obt[0][2], -0.8983139520017552, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.229802629928757, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.229802629928764, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.384394578043613, 6)

    def test_n3_diffuse_Ta(self):
        '''Tests EOM-CCSD(T)*a method.'''
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        '''
        # treating 1*1*2 supercell at gamma point
        supcell = super_cell(cell,nmp)
        gmf  = scf.GHF(supcell,exxdiv=None)
        ehf  = gmf.kernel()
        gcc  = cc.GCCSD(gmf)
        gcc.conv_tol=1e-12
        gcc.conv_tol_normt=1e-10
        gcc.max_cycle=250
        ecc, t1, t2 = gcc.kernel()
        print('GHF energy (supercell) %.7f \n' % (float(ehf)/2.))
        print('GCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))

        eom = eom_gccsd.EOMIP_Ta(gcc)
        e1, v = eom.ipccsd(nroots=6, koopmans=True)
        eom = eom_gccsd.EOMEA_Ta(gcc)
        e2, v = eom.eaccsd(nroots=6, koopmans=True)
        '''
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = cc.KGCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP_Ta(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.146351234409813, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.146351234404151, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.107255699646373, 6)

        eom = EOMEA_Ta(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.267728934041309, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.267728934041309, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.280954980102639, 6)

        eom = EOMEA_Ta(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.2290479727093149, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2290479727093468, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.384154366703175, 6)

