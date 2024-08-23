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
import numpy as np
from functools import reduce

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import gccsd
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd
from pyscf.cc import eom_gccsd

def setUpModule():
    global mol, mf, mol1, mf0, mf1, gmf, ucc, ucc0, ucc1
    global orbspin, nocca, noccb, nvira, nvirb, r1, r2
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-12)

    mol1 = mol.copy()
    mol1.spin = 2
    mol1.build()
    mf0 = scf.UHF(mol1).run(conv_tol=1e-12)
    mf1 = mf0.copy()

    nocca, noccb = mol1.nelec
    nmo = mol1.nao_nr()
    nvira, nvirb = nmo-nocca, nmo-noccb
    numpy.random.seed(12)
    mf1.mo_coeff = numpy.random.random((2,nmo,nmo)) - .5
    gmf = scf.addons.convert_to_ghf(mf1)
    orbspin = gmf.mo_coeff.orbspin

    ucc1 = cc.UCCSD(mf1)

    numpy.random.seed(11)
    no = nocca + noccb
    nv = nvira + nvirb
    r1 = numpy.random.random((no,nv)) - .9
    r2 = numpy.random.random((no,no,nv,nv)) - .9
    r2 = r2 - r2.transpose(1,0,2,3)
    r2 = r2 - r2.transpose(0,1,3,2)
    r1 = cc.addons.spin2spatial(r1, orbspin)
    r2 = cc.addons.spin2spatial(r2, orbspin)
    r1,r2 = eom_uccsd.vector_to_amplitudes_ee(
        eom_uccsd.amplitudes_to_vector_ee(r1,r2), ucc1.nmo, ucc1.nocc)
    ucc1.t1 = r1
    ucc1.t2 = r2

    ucc = cc.UCCSD(mf)
    ucc.max_space = 0
    ucc.conv_tol = 1e-8
    ucc.kernel()[0]

    ucc0 = cc.UCCSD(mf0)
    ucc0.conv_tol = 1e-8
    ucc0.direct = True
    ucc0.kernel()


def tearDownModule():
    global mol, mf, mol1, mf0, mf1, gmf, ucc, ucc0, ucc1
    del mol, mf, mol1, mf0, mf1, gmf, ucc, ucc0, ucc1

class KnownValues(unittest.TestCase):
    def test_ipccsd(self):
        eom = ucc.eomip_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.42789083399175043, 5)
        e,v = ucc.ipccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.42789083399175043, 5)
        self.assertAlmostEqual(e[2], 0.50226861340475437, 5)
        self.assertAlmostEqual(e[4], 0.68550641152952585, 5)

        e,v = ucc.ipccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.50226861340475437, 5)

    def test_ipccsd_koopmans(self):
        e,v = ucc.ipccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.42789083399175043, 5)
        self.assertAlmostEqual(e[2], 0.50226861340475437, 5)
        self.assertAlmostEqual(e[4], 0.68550641152952585, 5)

    def test_eaccsd(self):
        eom = ucc.eomea_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.19050592137699729, 5)
        e,v = ucc.eaccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.19050592137699729, 5)
        self.assertAlmostEqual(e[2], 0.28345228891172214, 5)
        self.assertAlmostEqual(e[4], 0.52280673926459342, 5)

        e,v = ucc.eaccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.28345228891172214, 5)

    def test_eaccsd_koopmans(self):
        e,v = ucc.eaccsd(nroots=6, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19050592137699729, 5)
        self.assertAlmostEqual(e[2], 0.28345228891172214, 5)
        self.assertAlmostEqual(e[4], 1.02136493172648370, 5)

        gcc1 = gccsd.GCCSD(scf.addons.convert_to_ghf(mf)).run()
        e1 = gcc1.eaccsd(nroots=6, koopmans=True)[0]
        self.assertAlmostEqual(abs(e1-e).max(), 0, 5)


    def test_eomee(self):
        self.assertAlmostEqual(ucc.e_corr, -0.13539788719099638, 5)
        eom = ucc.eomee_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.28114509667240556, 5)

        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[2], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[3], 0.30819728420902842, 5)

        e,v = ucc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.30819728420902842, 5)

    def test_eomee_ccsd_spin_keep(self):
        e, v = ucc.eomee_ccsd(nroots=2, koopmans=False)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[1], 0.30819728420902842, 5)

        e, v = ucc.eomee_ccsd(nroots=2, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[1], 0.30819728420902842, 5)

    def test_eomsf_ccsd(self):
        e, v = ucc.eomsf_ccsd(nroots=2, koopmans=False)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 5)

        e, v = ucc.eomsf_ccsd(nroots=2, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 5)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 5)

    def test_ucc_update_amps(self):
        gcc1 = gccsd.GCCSD(gmf)
        orbspin = gcc1._scf.mo_coeff.orbspin
        r1g = gcc1.spatial2spin(ucc1.t1, orbspin)
        r2g = gcc1.spatial2spin(ucc1.t2, orbspin)
        r1g, r2g = gcc1.update_amps(r1g, r2g, gcc1.ao2mo())
        u1g = gcc1.spin2spatial(r1g, orbspin)
        u2g = gcc1.spin2spatial(r2g, orbspin)
        eris1 = ucc1.ao2mo()
        t1, t2 = ucc1.update_amps(ucc1.t1, ucc1.t2, eris1)
        self.assertAlmostEqual(abs(u1g[0]-t1[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(u1g[1]-t1[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(u2g[0]-t2[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(u2g[1]-t2[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(u2g[2]-t2[2]).max(), 0, 5)
        self.assertAlmostEqual(float(abs(r1g-gcc1.spatial2spin(t1, orbspin)).max()), 0, 5)
        self.assertAlmostEqual(float(abs(r2g-gcc1.spatial2spin(t2, orbspin)).max()), 0, 5)
        self.assertAlmostEqual(uccsd.energy(ucc1, r1, r2, eris1), -7.2775115532675771, 8)
        e0, t1, t2 = ucc1.init_amps(eris1)
        self.assertAlmostEqual(lib.fp(cc.addons.spatial2spin(t1, orbspin)), 148.57054876656397, 8)
        self.assertAlmostEqual(lib.fp(cc.addons.spatial2spin(t2, orbspin)),-349.94207953071475, 8)
        self.assertAlmostEqual(e0, 30.640616265644827, 2)

    def test_ucc_eomee_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = [numpy.random.random((nocca,nvira))-.9,
              numpy.random.random((noccb,nvirb))-.9]
        r2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        r2[0] = r2[0] - r2[0].transpose(1,0,2,3)
        r2[0] = r2[0] - r2[0].transpose(0,1,3,2)
        r2[2] = r2[2] - r2[2].transpose(1,0,2,3)
        r2[2] = r2[2] - r2[2].transpose(0,1,3,2)

        gcc1 = cc.addons.convert_to_gccsd(ucc1)
        gr1 = gcc1.spatial2spin(r1)
        gr2 = gcc1.spatial2spin(r2)
        gee1 = eom_gccsd.EOMEE(gcc1)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)

        uee1 = eom_uccsd.EOMEESpinKeep(ucc1)
        orbspin = gcc1._scf.mo_coeff.orbspin
        r1 = uee1.spin2spatial(uee1.spatial2spin(r1, orbspin), orbspin)
        r2 = uee1.spin2spatial(uee1.spatial2spin(r2, orbspin), orbspin)
        vec = uee1.amplitudes_to_vector(r1,r2)
        vec1 = uee1.matvec(vec)

        r1, r2 = uee1.vector_to_amplitudes(vec1)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        self.assertAlmostEqual(float(abs(gr1-uee1.spatial2spin(r1, orbspin)).max()), 0, 9)
        self.assertAlmostEqual(float(abs(gr2-uee1.spatial2spin(r2, orbspin)).max()), 0, 9)
        self.assertAlmostEqual(lib.fp(vec1), 49.499911123484523, 9)

        ucc2 = ucc1.copy()
        ucc2.direct = True
        uee1 = eom_uccsd.EOMEESpinKeep(ucc2)
        vec1 = uee1.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), 49.499911123484523, 9)

    def test_ucc_eomee_ccsd_diag(self):
        vec1, vec2 = eom_uccsd.EOMEE(ucc1).get_diag()
        self.assertAlmostEqual(lib.fp(vec1), 62.767648620751018, 9)
        self.assertAlmostEqual(lib.fp(vec2), 156.2976365433517, 9)

        gcc1 = cc.addons.convert_to_gccsd(ucc1)
        gee1 = eom_gccsd.EOMEE(gcc1)
        gvec = gee1.get_diag()
        orbspin = gcc1._scf.mo_coeff.orbspin
        gv1, gv2 = gee1.vector_to_amplitudes(gvec)

        nmo = ucc1.nmo
        nocc = ucc1.nocc
        v1, v2 = eom_uccsd.vector_to_amplitudes_eomee(vec1, nmo, nocc)
        r1 = eom_uccsd.spin2spatial_eomee(gv1, orbspin)
        r2 = eom_uccsd.spin2spatial_eomee(gv2, orbspin)
        self.assertAlmostEqual(abs(abs(v1[0])-abs(r1[0])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v1[1])-abs(r1[1])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[0])-abs(r2[0])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[1])-abs(r2[1])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[2])-abs(r2[2])).max(), 0, 9)

        v1, v2 = eom_uccsd.vector_to_amplitudes_eomsf(vec2, nmo, nocc)
        r1 = eom_uccsd.spin2spatial_eomsf(gv1, orbspin)
        r2 = eom_uccsd.spin2spatial_eomsf(gv2, orbspin)
        self.assertAlmostEqual(abs(abs(v1[0])-abs(r1[0])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v1[1])-abs(r1[1])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[0])-abs(r2[0])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[1])-abs(r2[1])).max(), 0, 9)
        self.assertAlmostEqual(abs(abs(v2[2])-abs(r2[2])).max(), 0, 9)

    def test_ucc_eomee_init_guess(self):
        uee = eom_uccsd.EOMEESpinKeep(ucc1)
        diag = uee.get_diag()[0]
        guess = uee.get_init_guess(nroots=1, koopmans=False, diag=diag)
        self.assertAlmostEqual(lib.fp(guess[0]), -0.99525784369029358, 9)

        guess = uee.get_init_guess(nroots=1, koopmans=True, diag=diag)
        self.assertAlmostEqual(lib.fp(guess[0]), -0.84387013299273794, 9)

        guess = uee.get_init_guess(nroots=4, koopmans=False, diag=diag)
        self.assertAlmostEqual(lib.fp(guess), -0.98261980006133565, 9)

        guess = uee.get_init_guess(nroots=4, koopmans=True, diag=diag)
        self.assertAlmostEqual(lib.fp(guess), -0.38124032366955651, 9)

    def test_ucc_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEESpinFlip(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -1655.5564756993756, 8)

        gcc1 = cc.addons.convert_to_gccsd(ucc1)
        gee1 = eom_gccsd.EOMEE(gcc1)
        r1, r2 = myeom.vector_to_amplitudes(vec)
        orbspin = gcc1._scf.mo_coeff.orbspin
        r1 = myeom.spin2spatial(myeom.spatial2spin(r1, orbspin), orbspin)
        r2 = myeom.spin2spatial(myeom.spatial2spin(r2, orbspin), orbspin)
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)

        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(float(abs(gr1-myeom.spatial2spin(v1, orbspin)).max()), 0, 9)
        self.assertAlmostEqual(float(abs(gr2-myeom.spatial2spin(v2, orbspin)).max()), 0, 9)

        ucc2 = ucc1.copy()
        ucc2.direct = True
        myeom = eom_uccsd.EOMEESpinFlip(ucc2)
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -1655.5564756993756, 8)

    def test_ucc_eomip_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMIP(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -216.54704385639815, 8)

        gcc1 = cc.addons.convert_to_gccsd(ucc1)
        gee1 = eom_gccsd.EOMIP(gcc1)
        r1, r2 = myeom.vector_to_amplitudes(vec)
        orbspin = gcc1._scf.mo_coeff.orbspin
        r1 = myeom.spin2spatial(myeom.spatial2spin(r1, orbspin), orbspin)
        r2 = myeom.spin2spatial(myeom.spatial2spin(r2, orbspin), orbspin)
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)

        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(float(abs(gr1-myeom.spatial2spin(v1, orbspin)).max()), 0, 9)
        self.assertAlmostEqual(float(abs(gr2-myeom.spatial2spin(v2, orbspin)).max()), 0, 9)

        ucc2 = ucc1.copy()
        ucc2.direct = True
        myeom = eom_uccsd.EOMIP(ucc2)
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -216.54704385639815, 8)

    def test_ucc_eomea_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEA(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -110.5193188685705, 8)

        gcc1 = cc.addons.convert_to_gccsd(ucc1)
        gee1 = eom_gccsd.EOMEA(gcc1)
        r1, r2 = myeom.vector_to_amplitudes(vec)
        orbspin = gcc1._scf.mo_coeff.orbspin
        r1 = myeom.spin2spatial(myeom.spatial2spin(r1, orbspin), orbspin)
        r2 = myeom.spin2spatial(myeom.spatial2spin(r2, orbspin), orbspin)
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)

        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(float(abs(gr1-myeom.spatial2spin(v1, orbspin)).max()), 0, 9)
        self.assertAlmostEqual(float(abs(gr2-myeom.spatial2spin(v2, orbspin)).max()), 0, 9)

        ucc2 = ucc1.copy()
        ucc2.direct = True
        myeom = eom_uccsd.EOMEA(ucc2)
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -110.5193188685705, 8)

########################################
# With 4-fold symmetry in integrals
# max_memory = 0
# direct = True
    def test_eomee1(self):
        self.assertAlmostEqual(ucc0.e_corr, -0.10805861805688141, 5)
        e,v = ucc0.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0],-0.28757438579564343, 5)
        self.assertAlmostEqual(e[1], 7.0932490003970672e-05, 5)
        self.assertAlmostEqual(e[2], 0.026861582690761672, 5)
        self.assertAlmostEqual(e[3], 0.091111388761653589, 5)

        e,v = ucc0.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.091111388761653589, 5)

    def test_vector_to_amplitudes_eomsf(self):
        eomsf = eom_uccsd.EOMEESpinFlip(ucc0)
        size = eomsf.vector_size()
        v = numpy.random.random(size)
        r1, r2 = eomsf.vector_to_amplitudes(v)
        v1 = eomsf.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(v-v1).max(), 0, 12)

    def test_spatial2spin_eomsf(self):
        eomsf = eom_uccsd.EOMEESpinFlip(ucc0)
        size = eomsf.vector_size()
        v = numpy.random.random(size)
        r1, r2 = eomsf.vector_to_amplitudes(v)
        v1 = eomsf.spin2spatial(eomsf.spatial2spin(r1, orbspin), orbspin)
        v2 = eomsf.spin2spatial(eomsf.spatial2spin(r2, orbspin), orbspin)
        self.assertAlmostEqual(abs(r1[0]-v1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(r1[1]-v1[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[0]-v2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[1]-v2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[2]-v2[2]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[3]-v2[3]).max(), 0, 12)

    def test_vector_to_amplitudes_eom_spin_keep(self):
        eomsf = eom_uccsd.EOMEESpinKeep(ucc0)
        size = eomsf.vector_size()
        v = numpy.random.random(size)
        r1, r2 = eomsf.vector_to_amplitudes(v)
        v1 = eomsf.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(v-v1).max(), 0, 12)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.UHF(mol).apply(cc.UCCSD)
        nelec = (3,3)
        nocc = nelec
        nmo = (5, 5)
        mycc.nocc = nocc
        mycc.nmo = nmo
        def check_overwritten(method):
            vec = numpy.zeros(method.vector_size())
            vec_orig = vec.copy()
            ts = method.vector_to_amplitudes(vec)
            for ti in ts:
                if isinstance(ti, numpy.ndarray):
                    ti[:] = 1
                else:
                    for t in ti:
                        t[:] = 1
            self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

        check_overwritten(mycc)
        check_overwritten(mycc.EOMIP())
        check_overwritten(mycc.EOMEA())
        check_overwritten(mycc.EOMEESpinKeep())
        check_overwritten(mycc.EOMEESpinFlip())


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()
