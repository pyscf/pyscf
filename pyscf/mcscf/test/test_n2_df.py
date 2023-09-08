#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
import scipy.linalg
from pyscf import gto
from pyscf import scf
from pyscf import df
from pyscf import ao2mo
from pyscf import mcscf

def setUpModule():
    global mol, molsym, m, msym
    b = 1.4
    mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    max_memory = 1,
    )
    m = scf.RHF(mol)
    m.conv_tol = 1e-9
    m.scf()

    molsym = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
            ['N',(  0.000000,  0.000000, -b/2)],
            ['N',(  0.000000,  0.000000,  b/2)], ],
        basis = {'N': 'ccpvdz', },
        max_memory = 1,
        symmetry = True,
        )
    msym = scf.RHF(molsym)
    msym.conv_tol = 1e-9
    msym.scf()

def tearDownModule():
    global mol, molsym, m, msym
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym


class KnownValues(unittest.TestCase):
    def test_mc1step_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(m, 4, 4), auxbasis='weigend')
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(m, 4, 4), auxbasis='weigend')
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_4o4e_df(self):
        mc = mcscf.DFCASSCF(m, 4, 4, auxbasis='weigend')
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.9105231091045, 7)

    def test_mc2step_4o4e_df(self):
        mc = mcscf.density_fit(mcscf.CASSCF(m, 4, 4), auxbasis='weigend')
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.91052310869014, 7)

    def test_mc1step_6o6e_high_cost(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(m, 6, 6), auxbasis='weigend')
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_6o6e_high_cost(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(m, 6, 6), auxbasis='weigend')
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc1step_symm_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(msym, 4, 4), auxbasis='weigend')
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_symm_4o4e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(msym, 4, 4), auxbasis='weigend')
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_symm_6o6e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(msym, 6, 6), auxbasis='weigend')
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_symm_6o6e(self):
        mc = mcscf.approx_hessian(mcscf.CASSCF(msym, 6, 6), auxbasis='weigend')
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_casci_4o4e(self):
        mc = mcscf.CASCI(m.density_fit('weigend'), 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.88669369639578, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910276344981119, 4)

    def test_casci_symm_4o4e(self):
        mc = mcscf.CASCI(msym.density_fit('weigend'), 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.88669369639578, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910276344981119, 4)

    def test_casci_4o4e_1(self):
        mc = mcscf.DFCASCI(m.density_fit('weigend'), 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.88669369639578, 7)

    def test_casci_symm_4o4e_1(self):
        mc = mcscf.DFCASCI(msym.density_fit('weigend'), 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.88669369639578, 7)

    def test_casci_from_uhf(self):
        mf = scf.UHF(mol).run()
        mc = mcscf.CASCI(mf.density_fit('weigend'), 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.88669369639578, 6)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

    def test_casci_from_uhf1(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 4, 4))
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_df_ao2mo(self):
        mf = scf.density_fit(msym, auxbasis='weigend')
        mf.max_memory = 100
        mf.kernel()
        mc = mcscf.DFCASSCF(mf, 4, 4)
        with df.load(mf._cderi) as feri:
            cderi = numpy.asarray(feri)
        eri0 = numpy.dot(cderi.T, cderi)
        nmo = mc.mo_coeff.shape[1]
        ncore = mc.ncore
        nocc = ncore + mc.ncas
        eri0 = ao2mo.restore(1, ao2mo.kernel(eri0, mc.mo_coeff), nmo)
        eris = mc.ao2mo(mc.mo_coeff)
        self.assertTrue(numpy.allclose(eri0[:,:,ncore:nocc,ncore:nocc], eris.ppaa))
        self.assertTrue(numpy.allclose(eri0[:,ncore:nocc,:,ncore:nocc], eris.papa))

    def test_assign_cderi(self):
        nao = molsym.nao_nr()
        w, u = scipy.linalg.eigh(mol.intor('int2e_sph', aosym='s4'))
        idx = w > 1e-9

        mf = scf.density_fit(scf.RHF(molsym))
        mf._cderi = (u[:,idx] * numpy.sqrt(w[idx])).T.copy()
        mf.kernel()

        mc = mcscf.DFCASSCF(mf, 6, 6)
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -108.98010545803884, 7)

    def test_init(self):
        from pyscf.mcscf import df
        mf = scf.RHF(mol)
        self.assertTrue(isinstance(mcscf.CASCI(mf, 2, 2), mcscf.casci.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(mf.density_fit(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.CASCI(mf.newton(), 2, 2), mcscf.casci.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(mf.density_fit().newton(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.CASCI(mf.newton().density_fit(), 2, 2), mcscf.casci.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(mf.density_fit().newton().density_fit(), 2, 2), df._DFCASCI))

        self.assertTrue(isinstance(mcscf.CASSCF(mf, 2, 2), mcscf.mc1step.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(mf.density_fit(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(mf.newton(), 2, 2), mcscf.mc1step.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(mf.density_fit().newton(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(mf.newton().density_fit(), 2, 2), mcscf.mc1step.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(mf.density_fit().newton().density_fit(), 2, 2), df._DFCASSCF))

        self.assertTrue(isinstance(mcscf.DFCASCI(mf, 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.DFCASCI(mf.density_fit(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.DFCASCI(mf.newton(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.DFCASCI(mf.density_fit().newton(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.DFCASCI(mf.newton().density_fit(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.DFCASCI(mf.density_fit().newton().density_fit(), 2, 2), df._DFCASCI))

        self.assertTrue(isinstance(mcscf.DFCASSCF(mf, 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.DFCASSCF(mf.density_fit(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.DFCASSCF(mf.newton(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.DFCASSCF(mf.density_fit().newton(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.DFCASSCF(mf.newton().density_fit(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.DFCASSCF(mf.density_fit().newton().density_fit(), 2, 2), df._DFCASSCF))

        self.assertTrue(isinstance(mcscf.CASCI(msym, 2, 2), mcscf.casci_symm.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(msym.density_fit(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.CASCI(msym.newton(), 2, 2), mcscf.casci_symm.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(msym.density_fit().newton(), 2, 2), df._DFCASCI))
        self.assertTrue(isinstance(mcscf.CASCI(msym.newton().density_fit(), 2, 2), mcscf.casci_symm.CASCI))
        self.assertTrue(isinstance(mcscf.CASCI(msym.density_fit().newton().density_fit(), 2, 2), df._DFCASCI))

        self.assertTrue(isinstance(mcscf.CASSCF(msym, 2, 2), mcscf.mc1step_symm.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(msym.density_fit(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(msym.newton(), 2, 2), mcscf.mc1step_symm.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(msym.density_fit().newton(), 2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(msym.newton().density_fit(), 2, 2), mcscf.mc1step_symm.CASSCF))
        self.assertTrue(isinstance(mcscf.CASSCF(msym.density_fit().newton().density_fit(), 2, 2), df._DFCASSCF))

        self.assertTrue(isinstance(msym.CASCI(2, 2), mcscf.casci_symm.CASCI))
        self.assertTrue(isinstance(msym.density_fit().CASCI(2, 2), df._DFCASCI))
        self.assertTrue(isinstance(msym.density_fit().CASCI(2, 2), mcscf.casci_symm.CASCI))
        self.assertTrue(isinstance(msym.CASSCF(2, 2), mcscf.mc1step_symm.CASSCF))
        self.assertTrue(isinstance(msym.density_fit().CASSCF(2, 2), df._DFCASSCF))
        self.assertTrue(isinstance(msym.density_fit().CASSCF(2, 2), mcscf.mc1step_symm.CASSCF))


if __name__ == "__main__":
    print("Full Tests for density fitting N2")
    unittest.main()
