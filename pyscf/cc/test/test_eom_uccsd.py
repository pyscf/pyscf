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

import copy
import unittest
import numpy
from functools import reduce

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import gccsd
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd

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
mf1 = copy.copy(mf0)

nocca, noccb = mol1.nelec
nmo = mol1.nao_nr()
nvira, nvirb = nmo-nocca, nmo-noccb
numpy.random.seed(12)
mf1.mo_coeff = numpy.random.random((2,nmo,nmo)) - .5
gmf = scf.addons.convert_to_ghf(mf1)
orbspin = gmf.mo_coeff.orbspin

ucc1 = cc.UCCSD(mf1)
eris1 = ucc1.ao2mo()

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
ucc.conv_tol = 1e-10
ecc = ucc.kernel()[0]

ucc0 = cc.UCCSD(mf0)
ucc0.conv_tol = 1e-10
ucc0.direct = True
ucc0.kernel()

def tearDownModule():
    global mol, mf, mol1, mf0, mf1, gmf, ucc, ucc0, ucc1, eris1
    del mol, mf, mol1, mf0, mf1, gmf, ucc, ucc0, ucc1, eris1

class KnownValues(unittest.TestCase):
    def test_ipccsd(self):
        eom = ucc.eomip_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.42789083399175043, 6)
        e,v = ucc.ipccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.42789083399175043, 6)
        self.assertAlmostEqual(e[2], 0.50226861340475437, 6)
        self.assertAlmostEqual(e[4], 0.68550641152952585, 6)

        e,v = ucc.ipccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.50226861340475437, 6)

    def test_ipccsd_koopmans(self):
        e,v = ucc.ipccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.42789083399175043, 6)
        self.assertAlmostEqual(e[2], 0.50226861340475437, 6)
        self.assertAlmostEqual(e[4], 0.68550641152952585, 6)

    def test_eaccsd(self):
        eom = ucc.eomea_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.19050592137699729, 6)
        e,v = ucc.eaccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.19050592137699729, 6)
        self.assertAlmostEqual(e[2], 0.28345228891172214, 6)
        self.assertAlmostEqual(e[4], 0.52280673926459342, 6)

        e,v = ucc.eaccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.28345228891172214, 6)

    def test_eaccsd_koopmans(self):
        e,v = ucc.eaccsd(nroots=6, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19050592137699729, 6)
        self.assertAlmostEqual(e[2], 0.28345228891172214, 6)
        self.assertAlmostEqual(e[4], 1.02136493172648370, 6)

        gcc1 = gccsd.GCCSD(scf.addons.convert_to_ghf(mf)).run()
        e1 = gcc1.eaccsd(nroots=6, koopmans=True)[0]
        self.assertAlmostEqual(abs(e1-e).max(), 0, 6)


    def test_eomee(self):
        self.assertAlmostEqual(ecc, -0.13539788719099638, 6)
        eom = ucc.eomee_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.28114509667240556, 6)

        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[2], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[3], 0.30819728420902842, 6)

        e,v = ucc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.30819728420902842, 6)

    def test_eomee_ccsd_spin_keep(self):
        e, v = ucc.eomee_ccsd(nroots=2)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[1], 0.30819728420902842, 6)

        e, v = ucc.eomee_ccsd(nroots=2, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[1], 0.30819728420902842, 6)

    def test_eomsf_ccsd(self):
        e, v = ucc.eomsf_ccsd(nroots=2)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 6)

        e, v = ucc.eomsf_ccsd(nroots=2, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28114509667240556, 6)
        self.assertAlmostEqual(e[1], 0.28114509667240556, 6)

    def test_ucc_update_amps(self):
        gcc1 = gccsd.GCCSD(gmf)
        r1g = gcc1.spatial2spin(ucc1.t1, orbspin)
        r2g = gcc1.spatial2spin(ucc1.t2, orbspin)
        r1g, r2g = gcc1.update_amps(r1g, r2g, gcc1.ao2mo())
        u1g = gcc1.spin2spatial(r1g, orbspin)
        u2g = gcc1.spin2spatial(r2g, orbspin)
        t1, t2 = ucc1.update_amps(ucc1.t1, ucc1.t2, eris1)
        self.assertAlmostEqual(abs(u1g[0]-t1[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(u1g[1]-t1[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(u2g[0]-t2[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(u2g[1]-t2[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(u2g[2]-t2[2]).max(), 0, 6)
        self.assertAlmostEqual(float(abs(r1g-gcc1.spatial2spin(t1, orbspin)).max()), 0, 6)
        self.assertAlmostEqual(float(abs(r2g-gcc1.spatial2spin(t2, orbspin)).max()), 0, 6)
        self.assertAlmostEqual(uccsd.energy(ucc1, r1, r2, eris1), -7.2775115532675771, 8)
        e0, t1, t2 = ucc1.init_amps(eris1)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t1, orbspin)), 148.57054876656397, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t2, orbspin)),-349.94207953071475, 8)
        self.assertAlmostEqual(e0, 30.640616265644827, 2)

    def test_ucc_eomee_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r1 = cc.addons.spin2spatial(r1, orbspin)
        r2 = cc.addons.spin2spatial(r2, orbspin)
        vec = eom_uccsd.amplitudes_to_vector_ee(r1,r2)
        vec1 = eom_uccsd.eomee_ccsd_matvec(eom_uccsd.EOMEE(ucc1), vec)
        self.assertAlmostEqual(lib.finger(vec1), 275.11801889278121, 9)

    def test_ucc_eomee_ccsd_diag(self):
        vec1, vec2 = eom_uccsd.EOMEE(ucc1).get_diag()
        self.assertAlmostEqual(lib.finger(vec1),-36.776800901625307, 9)
        self.assertAlmostEqual(lib.finger(vec2), 106.70096636265369, 9)

    def test_ucc_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEESpinFlip(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = eom_uccsd.eomsf_ccsd_matvec(myeom, vec)
        self.assertAlmostEqual(lib.finger(vec1), -588.66159772500009, 8)

#    def test_ucc_eomip_matvec(self):
#
#    def test_ucc_eomea_matvec(self):

########################################
# With 4-fold symmetry in integrals
# max_memory = 0
# direct = True
    def test_eomee1(self):
        self.assertAlmostEqual(ucc0.e_corr, -0.10805861805688141, 6)
        e,v = ucc0.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0],-0.278489819261417580, 6)
        self.assertAlmostEqual(e[1], 0.004260352893874781, 6)
        self.assertAlmostEqual(e[2], 0.034029986071974654, 6)
        self.assertAlmostEqual(e[3], 0.091401096410169247, 6)

        e,v = ucc0.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.091401096410169247, 6)

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
        v1 = eom_uccsd.spin2spatial_eomsf(eom_uccsd.spatial2spin_eomsf(r1, orbspin), orbspin)
        v2 = eom_uccsd.spin2spatial_eomsf(eom_uccsd.spatial2spin_eomsf(r2, orbspin), orbspin)
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


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()

