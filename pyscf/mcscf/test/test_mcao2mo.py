#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf


class KnownValues(unittest.TestCase):
    def test_rhf(self):
        mol = gto.Mole()
        mol.verbose = 7
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
        eris2 = mcscf.mc_ao2mo._ERIS(mc, mo, 'outcore', level=1)
        eris3 = mcscf.mc_ao2mo._ERIS(mc, mo, 'outcore', level=2)
        self.assertTrue(numpy.allclose(eris0.vhf_c, eris1.vhf_c))
        self.assertTrue(numpy.allclose(eris0.j_pc , eris1.j_pc ))
        self.assertTrue(numpy.allclose(eris0.k_pc , eris1.k_pc ))
        self.assertTrue(numpy.allclose(eris0.ppaa , eris1.ppaa ))
        self.assertTrue(numpy.allclose(eris0.papa , eris1.papa ))

        self.assertTrue(numpy.allclose(eris0.vhf_c, eris2.vhf_c))
        self.assertTrue(numpy.allclose(eris0.j_pc , eris2.j_pc ))
        self.assertTrue(numpy.allclose(eris0.k_pc , eris2.k_pc ))
        self.assertTrue(numpy.allclose(eris0.ppaa , eris2.ppaa ))
        self.assertTrue(numpy.allclose(eris0.papa , eris2.papa ))

        self.assertTrue(numpy.allclose(eris0.vhf_c, eris3.vhf_c))
        self.assertTrue(numpy.allclose(eris0.ppaa , eris3.ppaa ))
        self.assertTrue(numpy.allclose(eris0.papa , eris3.papa ))

        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nmo = mo.shape[1]
        eri = ao2mo.incore.full(m._eri, mo, compact=False).reshape((nmo,)*4)
        aaap = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
        jc_pp = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
        kc_pp = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
        vhf_c = numpy.einsum('cij->ij', jc_pp)*2 - numpy.einsum('cij->ij', kc_pp)
        j_pc = numpy.einsum('ijj->ji', jc_pp)
        k_pc = numpy.einsum('ijj->ji', kc_pp)
        ppaa = numpy.array(eri[:,:,ncore:nocc,ncore:nocc])
        papa = numpy.array(eri[:,ncore:nocc,:,ncore:nocc])

        self.assertTrue(numpy.allclose(vhf_c, eris0.vhf_c))
        self.assertTrue(numpy.allclose(j_pc , eris0.j_pc ))
        self.assertTrue(numpy.allclose(k_pc , eris0.k_pc ))
        self.assertTrue(numpy.allclose(ppaa , eris0.ppaa ))
        self.assertTrue(numpy.allclose(papa , eris0.papa ))
        mol.stdout.close()

    def test_uhf(self):
        mol = gto.Mole()
        mol.verbose = 7
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

        mc = mcscf.umc1step.CASSCF(m, 4, 4)
        mc.verbose = 5
        mo = m.mo_coeff

        eris0 = mcscf.umc_ao2mo._ERIS(mc, mo, 'incore')
        eris1 = mcscf.umc_ao2mo._ERIS(mc, mo, 'outcore')
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
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for mc_ao2mo")
    unittest.main()
