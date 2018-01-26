import unittest
from pyscf import ao2mo
import numpy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.cc import rccsd
from pyscf.cc import addons
from pyscf.cc import rccsd_lambda
from pyscf.cc import gccsd, gccsd_lambda

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '631g'
mol.vebose = 5
mol.output = '/dev/null'
mol.build()
mf = scf.RHF(mol).run()
mycc = rccsd.RCCSD(mf)

class KnownValues(unittest.TestCase):
    def test_update_lambda_real(self):
        mycc = rccsd.RCCSD(mf)
        np.random.seed(12)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = np.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = np.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + np.diag(range(nmo))*2
        t1 = np.random.random((nocc,nvir))
        t2 = np.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = np.random.random((nocc,nvir))
        l2 = np.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)

        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        idx = np.tril_indices(nvir)
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = fock0

        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new, l2new = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(lib.finger(l1new), -6699.5335665027187, 9)
        self.assertAlmostEqual(lib.finger(l2new), -514.7001243502192 , 9)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).max(), 0, 12)

        mycc.max_memory = 0
        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new, l2new = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(lib.finger(l1new), -6699.5335665027187, 9)
        self.assertAlmostEqual(lib.finger(l2new), -514.7001243502192 , 9)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).max(), 0, 12)

    def test_update_lambda_complex(self):
        mo_coeff = mf.mo_coeff + np.sin(mf.mo_coeff) * .01j
        nao = mo_coeff.shape[0]
        eri = ao2mo.restore(1, mf._eri, nao)
        eri0 = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff.conj(), mo_coeff,
                          mo_coeff.conj(), mo_coeff)

        nocc, nvir = 5, nao-5
        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = np.diag(mf.mo_energy)

        np.random.seed(1)
        t1 = np.random.random((nocc,nvir)) + np.random.random((nocc,nvir))*.1j - .5
        t2 = np.random.random((nocc,nocc,nvir,nvir)) - .5
        t2 = t2 + np.sin(t2) * .1j
        t2 = t2 + t2.transpose(1,0,3,2)

        l1 = np.random.random((nocc,nvir)) + np.random.random((nocc,nvir))*.1j - .5
        l2 = np.random.random((nocc,nocc,nvir,nvir)) - .5
        l2 = l2 + np.sin(l2) * .1j
        l2 = l2 + l2.transpose(1,0,3,2)
        mycc = rccsd.RCCSD(mf)
        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new_ref, l2new_ref = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)

        orbspin = np.zeros(nao*2, dtype=int)
        orbspin[1::2] = 1
        eri1 = np.zeros([nao*2]*4, dtype=np.complex)
        eri1[0::2,0::2,0::2,0::2] = \
        eri1[0::2,0::2,1::2,1::2] = \
        eri1[1::2,1::2,0::2,0::2] = \
        eri1[1::2,1::2,1::2,1::2] = eri0
        eri1 = eri1.transpose(0,2,1,3) - eri1.transpose(0,2,3,1)
        erig = gccsd._PhysicistsERIs(mol)
        nocc *= 2
        nvir *= 2
        erig.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        erig.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        erig.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        erig.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        erig.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        erig.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        erig.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        mo_e = np.array([mf.mo_energy]*2)
        erig.fock = np.diag(mo_e.T.ravel())

        myccg = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        t1, t2 = myccg.amplitudes_from_ccsd(t1, t2)
        l1, l2 = myccg.amplitudes_from_ccsd(l1, l2)
        imds = gccsd_lambda.make_intermediates(myccg, t1, t2, erig)
        l1new, l2new = gccsd_lambda.update_lambda(myccg, t1, t2, l1, l2, erig, imds)
        self.assertAlmostEqual(abs(l1new[0::2,0::2]-l1new_ref).max(), 0, 9)
        l2aa = l2new[0::2,0::2,0::2,0::2]
        l2ab = l2new[0::2,1::2,0::2,1::2]
        self.assertAlmostEqual(abs(l2ab-l2new_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(l2ab-l2ab.transpose(1,0,2,3) - l2aa).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for RCCSD lambda")
    unittest.main()
