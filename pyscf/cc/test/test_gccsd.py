import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf.cc import gccsd
mol = gto.Mole()
mol.atom = [['O', (0.,   0., 0.)],
            ['O', (1.21, 0., 0.)]]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()
mf1 = scf.UHF(mol).run(conv_tol=1e-12)
mf1 = scf.addons.convert_to_ghf(mf1)

class KnownValues(unittest.TestCase):
    def test_ERIS(self):
        gcc = gccsd.GCCSD(mf1, frozen=4)
        mo_coeff = mf1.mo_coeff.copy()
        mo_coeff[-1,0] = 1e-12

        eris = gccsd._make_eris_incore(gcc, mf1.mo_coeff)
        self.assertAlmostEqual(lib.finger(eris.oooo),  0.61860222543125865, 12)
        self.assertAlmostEqual(lib.finger(eris.ooov), -0.57806300048513792, 12)
        self.assertAlmostEqual(lib.finger(eris.oovv), -0.93740596142345689, 12)
        self.assertAlmostEqual(lib.finger(eris.ovov), -1.3358817934823481 , 12)
        self.assertAlmostEqual(lib.finger(eris.ovvv),  8.9212684468927055 , 12)
        self.assertAlmostEqual(lib.finger(eris.vvvv),  9.3613744639720711 , 12)

        eris = gccsd._make_eris_incore(gcc, mo_coeff)
        self.assertAlmostEqual(lib.finger(eris.oooo),  0.61860222543125865, 12)
        self.assertAlmostEqual(lib.finger(eris.ooov), -0.57806300048513792, 12)
        self.assertAlmostEqual(lib.finger(eris.oovv), -0.93740596142345689, 12)
        self.assertAlmostEqual(lib.finger(eris.ovov), -1.3358817934823481 , 12)
        self.assertAlmostEqual(lib.finger(eris.ovvv),  8.9212684468927055 , 12)
        self.assertAlmostEqual(lib.finger(eris.vvvv),  9.3613744639720711 , 12)

        eris = gccsd._make_eris_outcore(gcc, mf1.mo_coeff)
        self.assertAlmostEqual(lib.finger(eris.oooo),  0.61860222543125865, 12)
        self.assertAlmostEqual(lib.finger(eris.ooov), -0.57806300048513792, 12)
        self.assertAlmostEqual(lib.finger(eris.oovv), -0.93740596142345689, 12)
        self.assertAlmostEqual(lib.finger(eris.ovov), -1.3358817934823481 , 12)
        self.assertAlmostEqual(lib.finger(eris.ovvv),  8.9212684468927055 , 12)
        self.assertAlmostEqual(lib.finger(eris.vvvv),  9.3613744639720711 , 12)

        eris = gccsd._make_eris_outcore(gcc, mo_coeff)
        self.assertAlmostEqual(lib.finger(eris.oooo),  0.61860222543125865, 12)
        self.assertAlmostEqual(lib.finger(eris.ooov), -0.57806300048513792, 12)
        self.assertAlmostEqual(lib.finger(eris.oovv), -0.93740596142345689, 12)
        self.assertAlmostEqual(lib.finger(eris.ovov), -1.3358817934823481 , 12)
        self.assertAlmostEqual(lib.finger(eris.ovvv),  8.9212684468927055 , 12)
        self.assertAlmostEqual(lib.finger(eris.vvvv),  9.3613744639720711 , 12)

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
        self.assertAlmostEqual(lib.finger(t1a), 13.702125264620541-302.62463606608304j, 11)
        self.assertAlmostEqual(lib.finger(t2a), -104.5636169484729+325.30707213752345j, 11)


if __name__ == "__main__":
    print("Full Tests for GCCSD")
    unittest.main()

