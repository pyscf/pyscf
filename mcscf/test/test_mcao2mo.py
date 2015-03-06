#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf


class KnowValues(unittest.TestCase):
    def test_rhf(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = 'cc-pvtz'
        mol.build()

        m = scf.RHF(mol)
        ehf = m.scf()
        mc = mcscf.CASSCF(m, 6, 4)
        mc.verbose = 5
        mo = m.mo_coeff

        eris0 = mcscf.mc_ao2mo._ERIS(mc, mo, 'incore')
        eris1 = mcscf.mc_ao2mo._ERIS(mc, mo, 'outcore')
        eris2 = mcscf.mc_ao2mo._ERIS(mc, mo, 'outcore', approx=1)
        eris3 = mcscf.mc_ao2mo._ERIS(mc, mo, 'outcore', approx=2)
        self.assertTrue(numpy.allclose(eris0.vhf_c, eris1.vhf_c))
        self.assertTrue(numpy.allclose(eris0.j_cp , eris1.j_cp ))
        self.assertTrue(numpy.allclose(eris0.k_cp , eris1.k_cp ))
        self.assertTrue(numpy.allclose(eris0.aapp , eris1.aapp ))
        self.assertTrue(numpy.allclose(eris0.appa , eris1.appa ))
        self.assertTrue(numpy.allclose(eris0.Iapcv, eris1.Iapcv))
        self.assertTrue(numpy.allclose(eris0.Icvcv, eris1.Icvcv))

        self.assertTrue(numpy.allclose(eris0.vhf_c, eris2.vhf_c))
        self.assertTrue(numpy.allclose(eris0.j_cp , eris2.j_cp ))
        self.assertTrue(numpy.allclose(eris0.k_cp , eris2.k_cp ))
        self.assertTrue(numpy.allclose(eris0.aapp , eris2.aapp ))
        self.assertTrue(numpy.allclose(eris0.appa , eris2.appa ))

        self.assertTrue(numpy.allclose(eris0.vhf_c, eris3.vhf_c))
        self.assertTrue(numpy.allclose(eris0.aapp , eris3.aapp ))
        self.assertTrue(numpy.allclose(eris0.appa , eris3.appa ))

        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nmo = mo.shape[1]
        eri = ao2mo.incore.full(m._eri, mo, compact=False).reshape((nmo,)*4)
        aaap = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
        jc_pp = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
        kc_pp = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
        vhf_c = numpy.einsum('cij->ij', jc_pp)*2 - numpy.einsum('cij->ij', kc_pp)
        j_cp = numpy.einsum('ijj->ij', jc_pp)
        k_cp = numpy.einsum('ijj->ij', kc_pp)
        aapp = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
        appa = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
        capv = eri[:ncore,ncore:nocc,:,ncore:]
        cvap = eri[:ncore,ncore:,ncore:nocc,:]
        cpav = eri[:ncore,:,ncore:nocc,ncore:]
        ccvv = eri[:ncore,:ncore,ncore:,ncore:]
        cvcv = eri[:ncore,ncore:,:ncore,ncore:]

        cVAp = cvap * 4 \
             - capv.transpose(0,3,1,2) \
             - cpav.transpose(0,3,2,1)
        cVCv = cvcv * 4 \
             - ccvv.transpose(0,3,1,2) \
             - cvcv.transpose(0,3,2,1)

        self.assertTrue(numpy.allclose(vhf_c, eris0.vhf_c))
        self.assertTrue(numpy.allclose(j_cp , eris0.j_cp ))
        self.assertTrue(numpy.allclose(k_cp , eris0.k_cp ))
        self.assertTrue(numpy.allclose(aapp , eris0.aapp ))
        self.assertTrue(numpy.allclose(appa , eris0.appa ))
        self.assertTrue(numpy.allclose(cVAp.transpose(2,3,0,1), eris1.Iapcv))
        self.assertTrue(numpy.allclose(cVCv.transpose(2,3,0,1), eris1.Icvcv))

    def test_uhf(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = 'cc-pvtz'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        m = scf.UHF(mol)
        ehf = m.scf()

        mc = mcscf.mc1step_uhf.CASSCF(m, 4, 4)
        mc.verbose = 5
        mo = m.mo_coeff

        eris0 = mcscf.mc_ao2mo_uhf._ERIS(mc, mo, 'incore')
        eris1 = mcscf.mc_ao2mo_uhf._ERIS(mc, mo, 'outcore')
        self.assertTrue(numpy.allclose(eris1.jkcpp, eris0.jkcpp))
        self.assertTrue(numpy.allclose(eris1.jkcPP, eris0.jkcPP))
        self.assertTrue(numpy.allclose(eris1.jC_pp, eris0.jC_pp))
        self.assertTrue(numpy.allclose(eris1.jc_PP, eris0.jc_PP))
        self.assertTrue(numpy.allclose(eris1.aapp , eris0.aapp ))
        self.assertTrue(numpy.allclose(eris1.aaPP , eris0.aaPP ))
        self.assertTrue(numpy.allclose(eris1.AApp , eris0.AApp ))
        self.assertTrue(numpy.allclose(eris1.AAPP , eris0.AAPP ))
        self.assertTrue(numpy.allclose(eris1.appa , eris0.appa ))
        self.assertTrue(numpy.allclose(eris1.apPA , eris0.apPA ))
        self.assertTrue(numpy.allclose(eris1.APPA , eris0.APPA ))
        self.assertTrue(numpy.allclose(eris1.cvCV , eris0.cvCV ))
        self.assertTrue(numpy.allclose(eris1.Icvcv, eris0.Icvcv))
        self.assertTrue(numpy.allclose(eris1.ICVCV, eris0.ICVCV))
        self.assertTrue(numpy.allclose(eris1.Iapcv, eris0.Iapcv))
        self.assertTrue(numpy.allclose(eris1.IAPCV, eris0.IAPCV))
        self.assertTrue(numpy.allclose(eris1.apCV , eris0.apCV ))
        self.assertTrue(numpy.allclose(eris1.APcv , eris0.APcv ))

        nmo = mo[0].shape[1]
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = (ncas + ncore[0], ncas + ncore[1])
        eriaa = ao2mo.incore.full(mc._scf._eri, mo[0])
        eriab = ao2mo.incore.general(mc._scf._eri, (mo[0],mo[0],mo[1],mo[1]))
        eribb = ao2mo.incore.full(mc._scf._eri, mo[1])
        eriaa = ao2mo.restore(1, eriaa, nmo)
        eriab = ao2mo.restore(1, eriab, nmo)
        eribb = ao2mo.restore(1, eribb, nmo)
        jkcpp = numpy.einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
              - numpy.einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
        jkcPP = numpy.einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
              - numpy.einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
        jC_pp = numpy.einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
        jc_PP = numpy.einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
        aapp = numpy.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        aaPP = numpy.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        AApp = numpy.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
        AAPP = numpy.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
        appa = numpy.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
        apPA = numpy.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
        APPA = numpy.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])

        cvCV = numpy.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
        Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
              - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
              - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
        ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
              - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
              - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)

        Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
              - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
              - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
        IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
              - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
              - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
        apCV = numpy.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
        APcv = numpy.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))

        self.assertTrue(numpy.allclose(jkcpp, eris0.jkcpp))
        self.assertTrue(numpy.allclose(jkcPP, eris0.jkcPP))
        self.assertTrue(numpy.allclose(jC_pp, eris0.jC_pp))
        self.assertTrue(numpy.allclose(jc_PP, eris0.jc_PP))
        self.assertTrue(numpy.allclose(aapp , eris0.aapp ))
        self.assertTrue(numpy.allclose(aaPP , eris0.aaPP ))
        self.assertTrue(numpy.allclose(AApp , eris0.AApp ))
        self.assertTrue(numpy.allclose(AAPP , eris0.AAPP ))
        self.assertTrue(numpy.allclose(appa , eris0.appa ))
        self.assertTrue(numpy.allclose(apPA , eris0.apPA ))
        self.assertTrue(numpy.allclose(APPA , eris0.APPA ))
        self.assertTrue(numpy.allclose(cvCV , eris0.cvCV ))
        self.assertTrue(numpy.allclose(Icvcv, eris0.Icvcv))
        self.assertTrue(numpy.allclose(ICVCV, eris0.ICVCV))
        self.assertTrue(numpy.allclose(Iapcv, eris0.Iapcv))
        self.assertTrue(numpy.allclose(IAPCV, eris0.IAPCV))
        self.assertTrue(numpy.allclose(apCV , eris0.apCV ))
        self.assertTrue(numpy.allclose(APcv , eris0.APcv ))


if __name__ == "__main__":
    print("Full Tests for mc_ao2mo")
    unittest.main()


