import make_test_cell
import numpy
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto
from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP, EOMEA
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP_Ta, EOMEA_Ta
from pyscf.pbc.lib import kpts_helper
from pyscf.cc import eom_uccsd
from pyscf.pbc.cc import kintermediates_rhf
from pyscf import lib
import unittest

cell_n3d = make_test_cell.test_cell_n3_diffuse()
kmf = pbcscf.KRHF(cell_n3d, cell_n3d.make_kpts((1,1,2), with_gamma_point=True), exxdiv=None)
kmf.conv_tol = 1e-10
kmf.scf()

# Helper functions
def kconserve_pmatrix(nkpts, kconserv):
    Ps = numpy.zeros((nkpts, nkpts, nkpts, nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                kb = kconserv[ki, ka, kj]
                Ps[ki, kj, ka, kb] = 1
    return Ps

def rand_t1_t2(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    numpy.random.seed(1)
    t1 = (numpy.random.random((nkpts, nocc, nvir)) +
          numpy.random.random((nkpts, nocc, nvir)) * 1j - .5 - .5j)
    t2 = (numpy.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) +
          numpy.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) * 1j - .5 - .5j)
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
    Ps = kconserve_pmatrix(nkpts, kconserv)
    t2 = t2 + numpy.einsum('xyzijab,xyzw->yxwjiba', t2, Ps)
    return t1, t2

def rand_r1_r2_ip(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    numpy.random.seed(1)
    r1 = (numpy.random.random((nocc,)) +
          numpy.random.random((nocc,)) * 1j - .5 - .5j)
    r2 = (numpy.random.random((nkpts, nkpts, nocc, nocc, nvir)) +
          numpy.random.random((nkpts, nkpts, nocc, nocc, nvir)) * 1j - .5 - .5j)
    return r1, r2

def rand_r1_r2_ea(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    numpy.random.seed(1)
    r1 = (numpy.random.random((nvir,)) +
          numpy.random.random((nvir,)) * 1j - .5 - .5j)
    r2 = (numpy.random.random((nkpts, nkpts, nocc, nvir, nvir)) +
          numpy.random.random((nkpts, nkpts, nocc, nvir, nvir)) * 1j - .5 - .5j)
    return r1, r2

def make_rand_kmf():
    # Not perfect way to generate a random mf.
    # CSC = 1 is not satisfied and the fock matrix is neither
    # diagonal nor sorted.
    numpy.random.seed(2)
    kmf = pbcscf.KRHF(cell_n3d, kpts=cell_n3d.make_kpts([1, 1, 3]))
    kmf.exxdiv = None
    nmo = cell_n3d.nao_nr()
    kmf.mo_occ = numpy.zeros((3, nmo))
    kmf.mo_occ[:, :2] = 2
    kmf.mo_energy = numpy.arange(nmo) + numpy.random.random((3, nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2
    kmf.mo_coeff = (numpy.random.random((3, nmo, nmo)) +
                    numpy.random.random((3, nmo, nmo)) * 1j - .5 - .5j)
    # Round to make this insensitive to small changes between PySCF versions
    mat_veff = kmf.get_veff().round(4)
    mat_hcore = kmf.get_hcore().round(4)
    kmf.get_veff = lambda *x: mat_veff
    kmf.get_hcore = lambda *x: mat_hcore
    return kmf

rand_kmf = make_rand_kmf()

def tearDownModule():
    global cell_n3d, kmf, rand_kmf
    del cell_n3d, kmf, rand_kmf

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
        e, v = myeom.eaccsd(nroots=2, left=True, koopmans=True, kptlist=(1,))
        self.assertAlmostEqual(e[0][0], 1.227583012965648, 6)
        self.assertAlmostEqual(e[0][1], 1.383037924670814, 6)
        e, v = myeom.eaccsd(nroots=2, left=True, koopmans=True, kptlist=(0,))
        self.assertAlmostEqual(e[0][0], 1.2669788599162801, 6)
        self.assertAlmostEqual(e[0][1], 1.2788832018377787, 6)

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
        eris.mo_energy = [eris.fock[ikpt].diagonal() for ikpt in range(cc.nkpts)]
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        eom = EOMIP_Ta(cc)
        e, v = eom.ipccsd(nroots=2, koopmans=True, kptlist=(0,), eris=eris)
        self.assertAlmostEqual(e[0][0], -1.146351230068405, 6)
        self.assertAlmostEqual(e[0][1], -1.10725570884212, 6)

        eom = EOMEA_Ta(cc)
        e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0], eris=eris)
        self.assertAlmostEqual(e[0][0], 1.267728933294929, 6)
        self.assertAlmostEqual(e[0][1], 1.280954973687476, 6)

        eom = EOMEA_Ta(cc)
        e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[1], eris=eris)
        self.assertAlmostEqual(e[0][0], 1.229047959680129, 6)
        self.assertAlmostEqual(e[0][1], 1.384154370672317, 6)

    def test_n3_diffuse_Ta_against_so(self):
        ehf_bench = -6.1870676561720721
        ecc_bench = -0.06764836939412185

        cc = pbcc.kccsd_rhf.RCCSD(kmf)
        cc.conv_tol = 1e-10
        eris = cc.ao2mo()
        eris.mo_energy = [eris.fock[ikpt].diagonal() for ikpt in range(cc.nkpts)]
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        eom = EOMEA_Ta(cc)
        eea_rccsd = eom.eaccsd_star(nroots=1, koopmans=True, kptlist=(0,), eris=eris)
        eom = EOMIP_Ta(cc)
        eip_rccsd = eom.ipccsd_star(nroots=1, koopmans=True, kptlist=(0,), eris=eris)
        self.assertAlmostEqual(eea_rccsd[0][0], 1.2610123166324307, 6)
        self.assertAlmostEqual(eip_rccsd[0][0], -1.1435100754903331, 6)

        from pyscf.pbc.cc import kccsd
        cc = pbcc.KGCCSD(kmf)
        cc.conv_tol = 1e-10
        eris = cc.ao2mo()
        eris.mo_energy = [eris.fock[ikpt].diagonal() for ikpt in range(cc.nkpts)]
        ecc, t1, t2 = cc.kernel(eris=eris)
        ehf = kmf.e_tot
        self.assertAlmostEqual(ehf, ehf_bench, 6)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

        from pyscf.pbc.cc import eom_kccsd_ghf
        eom = eom_kccsd_ghf.EOMEA_Ta(cc)
        eea_gccsd = eom.eaccsd_star(nroots=1, koopmans=True, kptlist=(0,), eris=eris)
        eom = eom_kccsd_ghf.EOMIP_Ta(cc)
        eip_gccsd = eom.ipccsd_star(nroots=1, koopmans=True, kptlist=(0,), eris=eris)
        self.assertAlmostEqual(eea_gccsd[0][0], 1.2610123166324307, 6)
        self.assertAlmostEqual(eip_gccsd[0][0], -1.1435100754903331, 6)

        # Usually slightly higher agreement when comparing directly against one another
        self.assertAlmostEqual(eea_gccsd[0][0], eea_rccsd[0][0], 9)
        self.assertAlmostEqual(eip_gccsd[0][0], eip_rccsd[0][0], 9)

    def test_t3p2_intermediates_complex(self):
        rand_cc = pbcc.kccsd_rhf.RCCSD(rand_kmf)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        e, pt1, pt2, Wmcik, Wacek = kintermediates_rhf.get_t3p2_imds_slow(rand_cc, t1, t2)
        self.assertAlmostEqual(lib.finger(e),47165803.3938429802656174+  0.0000000000000000j, 5)
        self.assertAlmostEqual(lib.finger(pt1),10444.3518376177471509+20016.3510856065695407j, 5)
        self.assertAlmostEqual(lib.finger(pt2),5481819.3905677245929837+-8012159.8432002812623978j, 5)
        self.assertAlmostEqual(lib.finger(Wmcik),-4401.1631306775143457+-10002.8851650238902948j, 5)
        self.assertAlmostEqual(lib.finger(Wacek),2057.9135114790879015+1970.9887693509299424j, 5)
