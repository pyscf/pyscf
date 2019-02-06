import make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto
from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP, EOMEA
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP_Ta, EOMEA_Ta
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
        nk = (1, 1, 2)
        ehf_bench = -6.1870676561720721
        ecc_bench = -0.06764836939412185

        cc = pbcc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol = 1e-9
        eris = cc.ao2mo()
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        e, v = cc.ipccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], -1.1489469942237946, 6)
        self.assertAlmostEqual(e[0][1], -1.1088194607458677, 6)
        e, v = cc.eaccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], 1.2669788600074476, 6)
        self.assertAlmostEqual(e[0][1], 1.278883198038047, 6)

        myeom = EOMIP(cc)
        imds = myeom.make_imds()
        e, v = myeom.ipccsd(nroots=2, koopmans=True, kptlist=(0,), imds=imds)
        self.assertAlmostEqual(e[0][0], -1.1489469942237946, 6)
        self.assertAlmostEqual(e[0][1], -1.1088194607458677, 6)
        e, v = myeom.ipccsd(nroots=2, koopmans=True, kptlist=(1,), imds=imds)
        self.assertAlmostEqual(e[0][0], -0.9074337254867506, 6)
        self.assertAlmostEqual(e[0][1], -0.9074331853695625, 6)
        e, v = myeom.ipccsd(nroots=2, left=True, koopmans=True, kptlist=(0,), imds=imds)
        self.assertAlmostEqual(e[0][0], -1.1489469931063192, 6)
        self.assertAlmostEqual(e[0][1], -1.1088194567671674, 6)
        e, v = myeom.ipccsd(nroots=2, left=True, koopmans=True, kptlist=(1,), imds=imds)
        self.assertAlmostEqual(e[0][0], -0.9074337234999493, 6)
        self.assertAlmostEqual(e[0][1], -0.9074331832202921, 6)

        myeom = EOMEA(cc)
        imds = myeom.make_imds()
        e, v = myeom.eaccsd(nroots=2, koopmans=True, kptlist=(1,))
        self.assertAlmostEqual(e[0][0], 1.2275830143478248, 6)
        self.assertAlmostEqual(e[0][1], 1.3830379248901867, 6)
        e, v = myeom.eaccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], 1.2669788600074476, 6)
        self.assertAlmostEqual(e[0][1], 1.278883198038047, 6)

    def test_n3_diffuse_frozen(self):
        nk = (1, 1, 2)
        ehf_bench = -6.1870676561720721
        ecc_bench = -0.0442506265840587

        cc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[0],[0,1]])
        cc.conv_tol = 1e-10
        eris = cc.ao2mo()
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        e, v = cc.ipccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], -1.1316152294295743, 6)
        self.assertAlmostEqual(e[0][1], -1.104163717600433, 6)
        e, v = cc.eaccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], 1.2572812499753756, 6)
        self.assertAlmostEqual(e[0][1], 1.280747357928012, 6)

        e, v = cc.ipccsd(nroots=2, koopmans=True, kptlist=(1,))
        self.assertAlmostEqual(e[0][0], -0.898314514845498, 6)
        self.assertAlmostEqual(e[0][1], -0.8983139526618168, 6)
        e, v = cc.eaccsd(nroots=2, koopmans=True, kptlist=(1,))
        self.assertAlmostEqual(e[0][0], 1.229802633498979, 6)
        self.assertAlmostEqual(e[0][1], 1.384394629885998, 6)

    def test_n3_diffuse_Ta(self):
        nk = (1, 1, 2)
        ehf_bench = -6.1870676561720721
        ecc_bench = -0.06764836939412185

        cc = pbcc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol = 1e-8
        eris = cc.ao2mo()
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        eom = EOMIP_Ta(cc)
        e, v = eom.ipccsd(nroots=2, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], -1.146351230068405, 6)
        self.assertAlmostEqual(e[0][1], -1.10725570884212, 6)

        eom = EOMEA_Ta(cc)
        e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0])
        self.assertAlmostEqual(e[0][0], 1.267728933294929, 6)
        self.assertAlmostEqual(e[0][1], 1.280954973687476, 6)

        eom = EOMEA_Ta(cc)
        e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e[0][0], 1.229047959680129, 6)
        self.assertAlmostEqual(e[0][1], 1.384154370672317, 6)

    def test_n3_diffuse_Ta_against_so(self):
        nk = (1, 1, 2)
        ehf_bench = -6.1870676561720721
        ecc_bench = -0.06764836939412185

        cc = pbcc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol = 1e-10
        eris = cc.ao2mo()
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        eom = EOMIP_Ta(cc)
        e, v = eom.ipccsd_star(nroots=2, koopmans=True, kptlist=(0,))
        #self.assertAlmostEqual(e[0][0], -1.146351230068405, 6)
        #self.assertAlmostEqual(e[0][1], -1.10725570884212, 6)

        from pyscf.pbc.cc import kccsd
        cc = pbcc.kccsd.KGCCSD(kmf)
        cc.conv_tol = 1e-10
        eris = cc.ao2mo()
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        from pyscf.pbc.cc import eom_kccsd_ghf
        eom = eom_kccsd_ghf.EOMIP_Ta(cc)
        e_ghf, v_ghf = eom.ipccsd_star(nroots=4, koopmans=True, kptlist=(0,), eris=eris)
        #self.assertAlmostEqual(e_ghf[0][0], -1.146351230068405, 6)
        #self.assertAlmostEqual(e_ghf[0][1], -1.10725570884212, 6)
        print "%20.16f" % e_ghf[0][0]
        print "%20.16f" % e[0][0]

        #eom = EOMEA_Ta(cc)
        #e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0])
        #self.assertAlmostEqual(e[0][0], 1.267728933294929, 6)
        #self.assertAlmostEqual(e[0][1], 1.280954973687476, 6)

        #eom = EOMEA_Ta(cc)
        #e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[1])
        #self.assertAlmostEqual(e[0][0], 1.229047959680129, 6)
        #self.assertAlmostEqual(e[0][1], 1.384154370672317, 6)
