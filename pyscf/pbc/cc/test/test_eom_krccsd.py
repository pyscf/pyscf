import copy
import pyscf.pbc.tools.make_test_cell as make_test_cell
import numpy
import numpy as np
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto
from pyscf.pbc import cc as pbcc
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP, EOMEA
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP_Ta, EOMEA_Ta
from pyscf.pbc.lib import kpts_helper
from pyscf.cc import eom_uccsd
from pyscf.pbc.cc import kintermediates, kintermediates_rhf
from pyscf import lib
import unittest

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

def make_rand_kmf(nkpts=3):
    # Not perfect way to generate a random mf.
    # CSC = 1 is not satisfied and the fock matrix is neither
    # diagonal nor sorted.
    numpy.random.seed(2)
    nkpts = nkpts
    kmf = pbcscf.KRHF(cell_n3d, kpts=cell_n3d.make_kpts([1, 1, nkpts]))
    kmf.exxdiv = None
    nmo = cell_n3d.nao_nr()
    kmf.mo_occ = numpy.zeros((nkpts, nmo))
    kmf.mo_occ[:, :2] = 2
    kmf.mo_energy = numpy.arange(nmo) + numpy.random.random((nkpts, nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2
    kmf.mo_coeff = (numpy.random.random((nkpts, nmo, nmo)) +
                    numpy.random.random((nkpts, nmo, nmo)) * 1j - .5 - .5j)
    ## Round to make this insensitive to small changes between PySCF versions
    #mat_veff = kmf.get_veff().round(4)
    #mat_hcore = kmf.get_hcore().round(4)
    #kmf.get_veff = lambda *x: mat_veff
    #kmf.get_hcore = lambda *x: mat_hcore
    return kmf

def setUpModule():
    global cell, cell_n3d, kmf, rand_kmf, rand_kmf1, rand_kmf2
    cell = gto.Cell()
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
    cell.precision = 1e-8
    cell.mesh = [15] * 3
    cell.build()

    cell_n3d = make_test_cell.test_cell_n3_diffuse()
    kmf = pbcscf.KRHF(cell_n3d, cell_n3d.make_kpts((1,1,2), with_gamma_point=True), exxdiv=None)
    kmf.conv_tol = 1e-10
    kmf.scf()

    rand_kmf = make_rand_kmf()
    rand_kmf1 = make_rand_kmf(nkpts=1)
    rand_kmf2 = make_rand_kmf(nkpts=2)

def tearDownModule():
    global cell, cell_n3d, kmf, rand_kmf, rand_kmf1, rand_kmf2
    cell_n3d.stdout.close()
    del cell, cell_n3d, kmf, rand_kmf, rand_kmf1, rand_kmf2

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
        eris.mo_energy = [eris.fock[ikpt].diagonal().real for ikpt in range(cc.nkpts)]
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
        eris.mo_energy = [eris.fock[ikpt].diagonal().real for ikpt in range(cc.nkpts)]
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
        eris.mo_energy = [eris.fock[ikpt].diagonal().real for ikpt in range(cc.nkpts)]
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
        self.assertAlmostEqual(eea_gccsd[0][0], eea_rccsd[0][0], 8)
        self.assertAlmostEqual(eip_gccsd[0][0], eip_rccsd[0][0], 8)

    def test_n3_ee(self):
        n = 15
        cell_n3 = make_test_cell.test_cell_n3([n]*3)
        kmf_n3 = pbcscf.KRHF(cell_n3, cell_n3.make_kpts([2,1,1]), exxdiv=None)
        kmf_n3.kernel()
        kmf_n3_ewald = pbcscf.KRHF(cell_n3, cell_n3.make_kpts([2,1,1]), exxdiv='ewald')
        kmf_n3_ewald.kernel()
        ehf_bench = [-8.651923514149, -10.530905169078]
        ecc_bench = [-0.155298299344, -0.093617975270]

        ekrhf = kmf_n3.e_tot
        self.assertAlmostEqual(ekrhf, ehf_bench[0], 3)
        ekrhf = kmf_n3_ewald.e_tot
        self.assertAlmostEqual(ekrhf, ehf_bench[1], 3)

        mycc = pbcc.KRCCSD(kmf_n3)
        ekrcc, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ekrcc, ecc_bench[0], 3)
        mycc_ewald = pbcc.KRCCSD(kmf_n3_ewald)
        mycc_ewald.keep_exxdiv = True
        ekrcc, t1, t2 = mycc_ewald.kernel()
        self.assertAlmostEqual(ekrcc, ecc_bench[1], 3)

        # EOM-EE-KRCCSD singlet
        from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
        nroots = 2  # number of roots requested

        myeomee = eom_krccsd.EOMEESinglet(mycc)
        myeomee.max_space = nroots * 10
        eee, vee = myeomee.kernel(nroots=nroots, kptlist=[0])
        self.assertAlmostEqual(eee[0][0], 0.267867075425, 3)
        self.assertAlmostEqual(eee[0][1], 0.268704338187, 3)
        eee, vee = myeomee.kernel(nroots=nroots, kptlist=[1])
        self.assertAlmostEqual(eee[0][0], 0.389795492091, 3)
        self.assertAlmostEqual(eee[0][1], 0.407782858154, 3)

        myeomee = eom_krccsd.EOMEESinglet(mycc_ewald)
        myeomee.max_space = nroots * 10
        eee, vee = myeomee.kernel(nroots=nroots, kptlist=[0])
        self.assertAlmostEqual(eee[0][0], 0.707047835495, 3)
        self.assertAlmostEqual(eee[0][1], 0.707047835495, 3)
        eee, vee = myeomee.kernel(nroots=nroots, kptlist=[1])
        self.assertAlmostEqual(eee[0][0], 0.815872164169, 3)
        self.assertAlmostEqual(eee[0][1], 0.845417271088, 3)

    def test_t3p2_imds_complex_slow(self):
        '''Test `_slow` t3p2 implementation.'''
        kmf = copy.copy(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        rand_cc = pbcc.kccsd_rhf.RCCSD(kmf)
        eris = rand_cc.ao2mo(kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        e, pt1, pt2, Wmcik, Wacek = kintermediates_rhf.get_t3p2_imds_slow(rand_cc, t1, t2)
        self.assertAlmostEqual(lib.fp(e),47165803.39384298, 2)
        self.assertAlmostEqual(lib.fp(pt1),10444.351837617747+20016.35108560657j, 4)
        self.assertAlmostEqual(lib.fp(pt2),5481819.3905677245929837+-8012159.8432002812623978j, 2)
        self.assertAlmostEqual(lib.fp(Wmcik),-4401.1631306775143457+-10002.8851650238902948j, 4)
        self.assertAlmostEqual(lib.fp(Wacek),2057.9135114790879015+1970.9887693509299424j, 4)

    def test_t3p2_imds_complex(self):
        '''Test t3p2 implementation.'''
        kmf = copy.copy(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        rand_cc = pbcc.kccsd_rhf.RCCSD(kmf)
        eris = rand_cc.ao2mo(kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        e, pt1, pt2, Wmcik, Wacek = kintermediates_rhf.get_t3p2_imds(rand_cc, t1, t2)
        self.assertAlmostEqual(lib.fp(e), 47165803.393840045, 2)
        self.assertAlmostEqual(lib.fp(pt1),10444.3518376177471509+20016.3510856065695407j, 4)
        self.assertAlmostEqual(lib.fp(pt2),5481819.3905677245929837+-8012159.8432002812623978j, 2)
        self.assertAlmostEqual(lib.fp(Wmcik),-4401.1631306775143457+-10002.8851650238902948j, 4)
        self.assertAlmostEqual(lib.fp(Wacek),2057.9135114790879015+1970.9887693509299424j, 4)

    def test_t3p2_imds_complex_against_so(self):
        '''Test t3[2] implementation against spin-orbital implmentation.'''
        from pyscf.pbc.scf.addons import convert_to_ghf
        kmf = copy.copy(rand_kmf2)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        rand_cc = pbcc.kccsd_rhf.RCCSD(kmf)
        eris = rand_cc.ao2mo(kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        e, pt1, pt2, Wmcik, Wacek = kintermediates_rhf.get_t3p2_imds(rand_cc, t1, t2)
        self.assertAlmostEqual(lib.fp(e), 3867.812511518491, 6)
        self.assertAlmostEqual(lib.fp(pt1),(179.0050003787795+521.7529255474592j), 6)
        self.assertAlmostEqual(lib.fp(pt2),(361.4902731606503+1079.5387279755082j), 6)
        self.assertAlmostEqual(lib.fp(Wmcik),(34.9811459194098-86.93467379996585j), 6)
        self.assertAlmostEqual(lib.fp(Wacek),(183.86684834783233+179.66583663669644j), 6)

        gkmf = convert_to_ghf(rand_kmf2)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = gkmf.get_veff().round(4)
        mat_hcore = gkmf.get_hcore().round(4)
        gkmf.get_veff = lambda *x: mat_veff
        gkmf.get_hcore = lambda *x: mat_hcore

        rand_gcc = pbcc.KGCCSD(gkmf)
        eris = rand_gcc.ao2mo(rand_gcc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_gcc.nkpts)]
        gt1 = rand_gcc.spatial2spin(t1)
        gt2 = rand_gcc.spatial2spin(t2)
        rand_gcc.t1, rand_gcc.t2, rand_gcc.eris = gt1, gt2, eris

        ge, gpt1, gpt2, gWmcik, gWacek = kintermediates.get_t3p2_imds_slow(rand_gcc, gt1, gt2)
        self.assertAlmostEqual(lib.fp(ge), lib.fp(e), 8)
        self.assertAlmostEqual(lib.fp(gpt1[:,::2,::2]), lib.fp(pt1), 8)
        self.assertAlmostEqual(lib.fp(gpt2[:,:,:,::2,1::2,::2,1::2]), lib.fp(pt2), 8)
        self.assertAlmostEqual(lib.fp(gWmcik[:,:,:,::2,1::2,::2,1::2]), lib.fp(Wmcik), 8)
        self.assertAlmostEqual(lib.fp(gWacek[:,:,:,::2,1::2,::2,1::2]), lib.fp(Wacek), 8)

    def test_t3p2_imds_complex_against_so_frozen(self):
        '''Test t3[2] implementation against spin-orbital implmentation with frozen orbitals.'''
        from pyscf.pbc.scf.addons import convert_to_ghf
        kmf = copy.copy(rand_kmf2)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        rand_cc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=1)
        eris = rand_cc.ao2mo(kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        e, pt1, pt2, Wmcik, Wacek = kintermediates_rhf.get_t3p2_imds(rand_cc, t1, t2)
        self.assertAlmostEqual(lib.fp(e), -328.44609187669454, 6)
        self.assertAlmostEqual(lib.fp(pt1),(-64.29455737653288+94.36604246905883j), 6)
        self.assertAlmostEqual(lib.fp(pt2),(-24.663592135920723+36.00181963359046j), 6)
        self.assertAlmostEqual(lib.fp(Wmcik),(6.692675632408793+6.926864923969868j), 6)
        self.assertAlmostEqual(lib.fp(Wacek),(24.78958393361647-15.627512899715132j), 6)

        gkmf = convert_to_ghf(rand_kmf2)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = gkmf.get_veff().round(4)
        mat_hcore = gkmf.get_hcore().round(4)
        gkmf.get_veff = lambda *x: mat_veff
        gkmf.get_hcore = lambda *x: mat_hcore

        rand_gcc = pbcc.KGCCSD(gkmf, frozen=2)
        eris = rand_gcc.ao2mo(rand_gcc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal().real for k in range(rand_gcc.nkpts)]
        gt1 = rand_gcc.spatial2spin(t1)
        gt2 = rand_gcc.spatial2spin(t2)
        rand_gcc.t1, rand_gcc.t2, rand_gcc.eris = gt1, gt2, eris

        ge, gpt1, gpt2, gWmcik, gWacek = kintermediates.get_t3p2_imds_slow(rand_gcc, gt1, gt2)
        self.assertAlmostEqual(lib.fp(ge), lib.fp(e), 8)
        self.assertAlmostEqual(lib.fp(gpt1[:,::2,::2]), lib.fp(pt1), 8)
        self.assertAlmostEqual(lib.fp(gpt2[:,:,:,::2,1::2,::2,1::2]), lib.fp(pt2), 8)
        self.assertAlmostEqual(lib.fp(gWmcik[:,:,:,::2,1::2,::2,1::2]), lib.fp(Wmcik), 8)
        self.assertAlmostEqual(lib.fp(gWacek[:,:,:,::2,1::2,::2,1::2]), lib.fp(Wacek), 8)

    def test_eomea_matvec(self):
        np.random.seed(2)
# Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
        nmo = cell.nao_nr()
        kmf.mo_occ = np.zeros((3, nmo))
        kmf.mo_occ[:, :2] = 2
        kmf.mo_energy = np.arange(nmo) + np.random.random((3, nmo)) * .3
        kmf.mo_energy[kmf.mo_occ == 0] += 2
        kmf.mo_coeff = (np.random.random((3, nmo, nmo)) +
                        np.random.random((3, nmo, nmo)) * 1j - .5 - .5j)

        mycc = pbcc.KRCCSD(kmf)
        t1, t2 = rand_t1_t2(kmf, mycc)
        mycc.t1 = t1
        mycc.t2 = t2

        eris = mycc.ao2mo()
        eom = EOMEA(mycc)
        imds = eom.make_imds(eris)
        np.random.seed(9)
        vector = np.random.random(eom.vector_size())

        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-2.615041322934018 -0.19907655222705176j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9105694363906784+0.4623840337230889j ), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.5191624937262938-0.09803982911194647j), 7)

        kmf = kmf.density_fit(auxbasis=[[0, (2., 1.)], [0, (1., 1.)], [0, (.5, 1.)]])
        mycc._scf = kmf

        mycc.max_memory = 0
        eris = mycc.ao2mo()
        imds = eom.make_imds(eris)
        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-2.6242967982318532-0.19622574939883755j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9052161075024587+0.4635723967077203j ), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.5273812229833275-0.10165584293391894j), 7)

        mycc.max_memory = 4000
        eris = mycc.ao2mo()
        imds = eom.make_imds(eris)
        hc = eom.matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-2.6242967982318532-0.19622574939883755j), 7)
        hc = eom.matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9052161075024587+0.4635723967077203j ), 7)
        hc = eom.matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-3.5273812229833275-0.10165584293391894j), 7)

    def test_eomea_l_matvec(self):
        np.random.seed(2)
# Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
        nmo = cell.nao_nr()
        kmf.mo_occ = np.zeros((3, nmo))
        kmf.mo_occ[:, :2] = 2
        kmf.mo_energy = np.arange(nmo) + np.random.random((3, nmo)) * .3
        kmf.mo_energy[kmf.mo_occ == 0] += 2
        kmf.mo_coeff = (np.random.random((3, nmo, nmo)) +
                        np.random.random((3, nmo, nmo)) * 1j - .5 - .5j)

        mycc = pbcc.KRCCSD(kmf)
        t1, t2 = rand_t1_t2(kmf, mycc)
        mycc.t1 = t1
        mycc.t2 = t2

        eris = mycc.ao2mo()
        eom = EOMEA(mycc)
        imds = eom.make_imds(eris)
        np.random.seed(9)
        vector = np.random.random(eom.vector_size())

        hc = eom.l_matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.9490117387531858-1.726564412656459j), 7)
        hc = eom.l_matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.4497554439273588-5.620765390422395j), 7)
        hc = eom.l_matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9057184472068758+2.7776122802218817j), 7)

        kmf = kmf.density_fit(auxbasis=[[0, (2., 1.)], [0, (1., 1.)], [0, (.5, 1.)]])
        mycc._scf = kmf
        mycc.max_memory = 0
        eris = mycc.ao2mo()
        imds = eom.make_imds(eris)
        hc = eom.l_matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.9525095721066594-1.722602584395692j), 7)
        hc = eom.l_matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.4402079681364959-5.610500177034039j), 7)
        hc = eom.l_matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9053243731138183+2.785112360342188j), 7)

        mycc.max_memory = 4000
        eris = mycc.ao2mo()

        imds = eom.make_imds(eris)
        hc = eom.l_matvec(vector, 0, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.9525095721066594-1.722602584395692j), 7)
        hc = eom.l_matvec(vector, 1, imds)
        self.assertAlmostEqual(lib.fp(hc), (-0.4402079681364959-5.610500177034039j), 7)
        hc = eom.l_matvec(vector, 2, imds)
        self.assertAlmostEqual(lib.fp(hc), (-1.9053243731138183+2.785112360342188j), 7)

if __name__ == '__main__':
    print("eom_kccsd_rhf tests")
    unittest.main()
