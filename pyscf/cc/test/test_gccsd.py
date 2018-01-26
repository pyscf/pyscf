import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf.cc import gccsd
from pyscf import ao2mo
from pyscf.cc import gccsd_rdm
from pyscf.cc import uccsd
from pyscf.cc import addons

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.verbose = 5
mol.output = '/dev/null'
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()
mf1 = scf.UHF(mol).run(conv_tol=1e-12)
mf1 = scf.addons.convert_to_ghf(mf1)

gcc1 = gccsd.GCCSD(mf1).run(conv_tol=1e-9)

class KnownValues(unittest.TestCase):
    def test_gccsd(self):
        self.assertAlmostEqual(gcc1.e_corr, -0.18212844850615587, 7)

    def test_ERIS(self):
        gcc = gccsd.GCCSD(mf1, frozen=4)
        numpy.random.seed(9)
        mo_coeff0 = numpy.random.random(mf1.mo_coeff.shape) - .9
        mo_coeff1 = mo_coeff0.copy()
        mo_coeff1[-1,0] = 1e-12

        eris = gccsd._make_eris_incore(gcc, mo_coeff0)
        self.assertAlmostEqual(lib.finger(eris.oooo), -274.88757393088122, 9)
        self.assertAlmostEqual(lib.finger(eris.ooov),  1058.7275352964637, 9)
        self.assertAlmostEqual(lib.finger(eris.oovv), -160.01426275786184, 9)
        self.assertAlmostEqual(lib.finger(eris.ovov), -8691.5932754980749, 9)
        self.assertAlmostEqual(lib.finger(eris.ovvv), -549.93089259191584, 9)
        self.assertAlmostEqual(lib.finger(eris.vvvv), -2171.9867436901673, 9)

        eris = gccsd._make_eris_incore(gcc, mo_coeff1)
        self.assertAlmostEqual(lib.finger(eris.oooo), -274.88757393088122, 9)
        self.assertAlmostEqual(lib.finger(eris.ooov),  1058.7275352964637, 9)
        self.assertAlmostEqual(lib.finger(eris.oovv), -160.01426275786184, 9)
        self.assertAlmostEqual(lib.finger(eris.ovov), -8691.5932754980749, 9)
        self.assertAlmostEqual(lib.finger(eris.ovvv), -549.93089259191584, 9)
        self.assertAlmostEqual(lib.finger(eris.vvvv), -2171.9867436901673, 9)

        eris = gccsd._make_eris_outcore(gcc, mo_coeff0)
        self.assertAlmostEqual(lib.finger(eris.oooo), -274.88757393088122, 9)
        self.assertAlmostEqual(lib.finger(eris.ooov),  1058.7275352964637, 9)
        self.assertAlmostEqual(lib.finger(eris.oovv), -160.01426275786184, 9)
        self.assertAlmostEqual(lib.finger(eris.ovov), -8691.5932754980749, 9)
        self.assertAlmostEqual(lib.finger(eris.ovvv), -549.93089259191584, 9)
        self.assertAlmostEqual(lib.finger(eris.vvvv), -2171.9867436901673, 9)

        eris = gccsd._make_eris_outcore(gcc, mo_coeff1)
        self.assertAlmostEqual(lib.finger(eris.oooo), -274.88757393088122, 9)
        self.assertAlmostEqual(lib.finger(eris.ooov),  1058.7275352964637, 9)
        self.assertAlmostEqual(lib.finger(eris.oovv), -160.01426275786184, 9)
        self.assertAlmostEqual(lib.finger(eris.ovov), -8691.5932754980749, 9)
        self.assertAlmostEqual(lib.finger(eris.ovvv), -549.93089259191584, 9)
        self.assertAlmostEqual(lib.finger(eris.vvvv), -2171.9867436901673, 9)

    def test_update_amps(self):
        mol = gto.M()
        nocc, nvir = 8, 14
        nmo = nocc + nvir
        nmo_pair = nmo*(nmo+1)//2
        mf = scf.GHF(mol)
        numpy.random.seed(12)
        mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
        mf.mo_coeff = numpy.random.random((nmo,nmo))
        mf.mo_energy = numpy.arange(0., nmo)
        mf.mo_occ = numpy.zeros(nmo)
        mf.mo_occ[:nocc] = 1
        vhf = numpy.random.random((nmo,nmo)) + numpy.random.random((nmo,nmo))+1j
        vhf = vhf + vhf.conj().T
        mf.get_veff = lambda *args: vhf
        cinv = numpy.linalg.inv(mf.mo_coeff)
        mf.get_hcore = lambda *args: (reduce(numpy.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
        nmo_pair = nmo*(nmo//2+1)//4
        mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
        mycc = gccsd.GCCSD(mf)
        eris = mycc.ao2mo()
        eris.oooo = eris.oooo + numpy.sin(eris.oooo)*1j
        eris.oooo = eris.oooo + eris.oooo.conj().transpose(2,3,0,1)
        eris.ooov = eris.ooov + numpy.sin(eris.ooov)*1j
        eris.oovv = eris.oovv + numpy.sin(eris.oovv)*1j
        eris.ovov = eris.ovov + numpy.sin(eris.ovov)*1j
        eris.ovov = eris.ovov + eris.ovov.conj().transpose(2,3,0,1)
        eris.ovvv = eris.ovvv + numpy.sin(eris.ovvv)*1j
        eris.vvvv = eris.vvvv + numpy.sin(eris.vvvv)*1j
        eris.vvvv = eris.vvvv + eris.vvvv.conj().transpose(2,3,0,1)
        a = numpy.random.random((nmo,nmo)) * .1
        eris.fock += a + a.T
        t1 = numpy.random.random((nocc,nvir))*.1 + numpy.random.random((nocc,nvir))*.1j
        t2 = (numpy.random.random((nocc,nocc,nvir,nvir))*.1 +
              numpy.random.random((nocc,nocc,nvir,nvir))*.1j)
        t2 = t2 - t2.transpose(0,1,3,2)
        t2 = t2 - t2.transpose(1,0,2,3)
        r1, r2 = mycc.vector_to_amplitudes(mycc.amplitudes_to_vector(t1, t2))
        self.assertAlmostEqual(abs(t1-r1).max(), 0, 14)
        self.assertAlmostEqual(abs(t2-r2).max(), 0, 14)

        t1a, t2a = mycc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(lib.finger(t1a), 20.805393111419136-300.1138026015621j, 9)
        self.assertAlmostEqual(lib.finger(t2a),-313.54117398035567+8.3700078645035205j, 9)

    def test_rdm(self):
        nocc = 6
        nvir = 10
        mf = scf.GHF(mol)
        nmo = nocc + nvir
        npair = nmo*(nmo//2+1)//4
        numpy.random.seed(12)
        mf._eri = numpy.random.random(npair*(npair+1)//2)*.3
        hcore = numpy.random.random((nmo,nmo)) * .5
        hcore = hcore + hcore.T + numpy.diag(range(nmo))*2
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        mf.mo_coeff = numpy.eye(nmo)
        mf.mo_occ = numpy.zeros(nmo)
        mf.mo_occ[:nocc] = 1
        dm1 = mf.make_rdm1()
        mf.e_tot = mf.energy_elec()[0]
        mycc = gccsd.GCCSD(mf)
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        dm1 = gccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
        dm2 = gccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2)
        nao = nmo // 2
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        eri  = ao2mo.kernel(mf._eri, mo_a)
        eri += ao2mo.kernel(mf._eri, mo_b)
        eri1 = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b))
        eri += eri1
        eri += eri1.T
        eri = ao2mo.restore(1, eri, nmo)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,jilk', eri, dm2) * .5
        self.assertAlmostEqual(e1, mycc.e_tot, 7)

        def antisym(t2):
            t2 = t2 - t2.transpose(0,1,3,2)
            t2 = t2 - t2.transpose(1,0,2,3)
            return t2
        numpy.random.seed(1)
        t1 = numpy.random.random((2,nocc,nvir))*.1 - .1
        t2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
        t2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        t2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        t2 = (t2aa,t2ab,t2bb)
        t1 = addons.spatial2spin(t1)
        t2 = addons.spatial2spin(t2)
        l1 = numpy.random.random((2,nocc,nvir))*.1 - .1
        l2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
        l2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        l2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        l2 = (l2aa,l2ab,l2bb)
        l1 = addons.spatial2spin(l1)
        l2 = addons.spatial2spin(l2)

        dm1 = gccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
        dm2 = gccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2).conj()).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(2,1,0,3)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(0,3,2,1)       ).max(), 0, 9)

    def test_h2o_rdm(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.basis = '631g'
        mol.spin = 2
        mol.build()
        mf = scf.UHF(mol).run()
        myucc = uccsd.UCCSD(mf).run()
        udm1 = myucc.make_rdm1()
        udm2 = myucc.make_rdm2()

        mf = scf.addons.convert_to_ghf(mf)
        mycc = gccsd.GCCSD(mf)
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        dm1 = gccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
        dm2 = gccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2)

        nao = mol.nao_nr()
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        nmo = mo_a.shape[1]
        eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
        orbspin = mf.mo_coeff.orbspin
        sym_forbid = (orbspin[:,None] != orbspin)
        eri[sym_forbid,:,:] = 0
        eri[:,:,sym_forbid] = 0
        hcore = scf.RHF(mol).get_hcore()
        h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,jilk', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 7)

        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        self.assertAlmostEqual(abs(dm1[idxa[:,None],idxa] - udm1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(dm1[idxb[:,None],idxb] - udm1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa] - udm2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb] - udm2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb] - udm2[2]).max(), 0, 9)


if __name__ == "__main__":
    print("Tests for GCCSD")
    unittest.main()

