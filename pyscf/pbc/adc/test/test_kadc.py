import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, adc
import unittest

def setUpModule():
    global cell, kmf, mycc, eris
    cell = make_test_cell.test_cell_n3_diffuse()
    kmf = scf.KRHF(cell, kpts=cell.make_kpts(
        [1,1,2], with_gamma_point=True), exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.conv_tol_grad = 1e-6
    kmf.verbose = 0
    kmf.scf()

def tearDownModule():
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):

        # Running HF and MP2 and MP3 with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1729762695142645, 4)

        myadc  = adc.KRADC(kmf)
        e_mp2, t1, t2 = myadc.kernel_gs()
        self.assertAlmostEqual(e_mp2, -0.11464808675651168, 4)

        # IP-ADC(2)
        myadc.method = 'adc(2)'
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[0])
        self.assertAlmostEqual(e[0][0],-1.1684386160049567,4)
        self.assertAlmostEqual(e[0][1],-1.0944210679002468,4)
        self.assertAlmostEqual(e[0][2],-1.0944210623845938,4)

        # Ensure kptlist behaves correctly
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[1])
        self.assertAlmostEqual(e[0][0], -1.0362468821854942, 4)
        self.assertAlmostEqual(e[0][1], -1.0015050525294895, 4)
        self.assertAlmostEqual(e[0][2], -1.0015050525294888, 4)

        # IP-ADC(3)
        myadc.method = 'adc(3)'
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[0])
        self.assertAlmostEqual(e[0][0],-1.1740951190044178, 4)
        self.assertAlmostEqual(e[0][1],-1.1536065878793587, 4)
        self.assertAlmostEqual(e[0][2],-1.153606584783398, 4)

if __name__ == '__main__':
    print("kadc_ip_basic tests")
    unittest.main()
