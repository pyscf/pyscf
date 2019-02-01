import make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto
from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.eom_kccsd_uhf import EOMIP, EOMEA
from pyscf.cc import eom_uccsd
import unittest

cell_n3d = make_test_cell.test_cell_n3_diffuse()
kmf = pbcscf.KRHF(cell_n3d, cell_n3d.make_kpts((1,1,2), with_gamma_point=True), exxdiv=None)
kmf.conv_tol = 1e-10
kmf.scf()

def tearDownModule():
    global cell_n3d, kmf
    del cell_n3d, kmf

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):
        self.assertAlmostEqual(kmf.e_tot, -6.1870676561721227, 6)
        cell = make_test_cell.test_cell_n3_diffuse()
        nmp = [1,1,2]
        '''
        # treating 1*1*2 supercell at gamma point
        supcell = super_cell(cell,nmp)
        gmf  = scf.UHF(supcell,exxdiv=None)
        ehf  = gmf.kernel()
        gcc  = cc.UCCSD(gmf)
        gcc.conv_tol=1e-12
        gcc.conv_tol_normt=1e-10
        gcc.max_cycle=250
        ecc, t1, t2 = gcc.kernel()
        print('UHF energy (supercell) %.7f \n' % (float(ehf)/2.))
        print('UCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))
        #eom = eom_uccsd.EOMIP(gcc)
        #e1, v = eom.ipccsd(nroots=2)
        #eom = eom_uccsd.EOMEA(gcc)
        #e2, v = eom.eaccsd(nroots=2, koopmans=True)
        '''
        mycc = pbcc.KUCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0676483711263548, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, kptlist=[0])
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
        self.assertAlmostEqual(e2_obt[0][0], 1.227583017460536, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.227583017460617, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.383037918699404, 6)

    def test_n3_diffuse_frozen(self):
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = pbcc.KUCCSD(kmf, frozen=([[0,],[0,1]],[[0],[0,1]]))
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0442506265840587, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.1316152294295743, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1041637212148683, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.104163717600433, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.2572812499753756, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.257281253091707, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.2807473549827182, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=2, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e1_obt[0][0], -0.8983145129187627, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.8983145129187627, 6)
        #self.assertAlmostEqual(e1_obt[0][2], -0.09623848794144876, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.229802629928757, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.229802629928764, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.384394578043613, 6)
