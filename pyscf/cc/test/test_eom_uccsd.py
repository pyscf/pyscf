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
mol.basis = 'cc-pvdz'
mol.spin = 0
mol.build()
mf = scf.UHF(mol).run(conv_tol=1e-12)

mol1 = mol.copy()
mol1.spin = 2
mol1.build()
mf1 = scf.UHF(mol1).run(conv_tol=1e-12)

nocca, noccb = mol1.nelec
nvira, nvirb = 18, 20
nmo = mol1.nao_nr()
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
ucc.conv_tol = 1e-10
ecc, t1, t2 = ucc.kernel()

ucc0 = cc.UCCSD(mf)
ucc0.conv_tol = 1e-10
ucc0.max_space = 0
ucc0.direct = True
ucc0.kernel()

class KnownValues(unittest.TestCase):
    def test_frozen(self):
        mf1 = scf.UHF(mol1).run()
        # Freeze 1s electrons
        frozen = [[0,1], [0,1]]
        ucc = cc.UCCSD(mf1, frozen=frozen)
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.12065952206237093, 8)

    def test_ipccsd(self):
        e,v = ucc.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 6)
        e,v = ucc.ipccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[4], 0.6782876002229172, 6)

        e,v = ucc.ipccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)

    def test_ipccsd_koopmans(self):
        e,v = ucc.ipccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[4], 0.6782876002229172, 6)

    def test_eaccsd(self):
        e,v = ucc.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.16737886338859731, 6)
        e,v = ucc.eaccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[4], 0.51006797826488071, 6)

        e,v = ucc.eaccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)

    def test_eaccsd_koopmans(self):
        e,v = ucc.eaccsd(nroots=6, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[4], 0.73443353474355455, 6)

        gcc1 = gccsd.GCCSD(scf.addons.convert_to_ghf(mf)).run()
        e1 = gcc1.eaccsd(nroots=6, koopmans=True)[0]
        self.assertAlmostEqual(abs(e1-e).max(), 0, 6)


    def test_eomee(self):
        self.assertAlmostEqual(ecc, -0.2133432430989155, 6)
        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = ucc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)


    def test_ucc_eris(self):
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.oooo)),-10.916243719485539 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovoo)), 2.260901088440082  , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovov)),-1.5026561669936918 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.oovv)), 80.195494913302227 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovvo)), 1.2041503835528897 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovvv)), 2.5596389386539222 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.vvvv)), 23.034764419245832 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOOO)), 3.4295150561303527 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVOO)),-6.6125506082923291 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVOV)),-0.17635855210079643, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOVV)),-3.4163309360534324 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVVO)),-1.1325213672960015 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVVV)),-35.7258983802149   , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.VVVV)),-56.05285722390785  , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ooOO)), 3.0648586544587877 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovOO)), 3.3617389830062985 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovOV)), 11.854191316348441 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ooVV)),-50.382870054040161 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovVO)),-11.614143609895805 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovVV)),-29.354475449200962 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.vvVV)), 24.742886959540034 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVoo)),-4.2266948384045566 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOvv)), 13.492853848604197 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVvo)), 12.548800971909802 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVvv)), 0.82841253863142228, 9)

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
        self.assertAlmostEqual(abs(r1g-gcc1.spatial2spin(t1, orbspin)).max(), 0, 6)
        self.assertAlmostEqual(abs(r2g-gcc1.spatial2spin(t2, orbspin)).max(), 0, 6)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t1, orbspin)), -22.760744706991609, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t2, orbspin)), -10542.613510254763, 5)
        self.assertAlmostEqual(uccsd.energy(ucc1, r1, r2, eris1), -12.258116633736986, 8)
        e0, t1, t2 = ucc1.init_amps(eris1)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t1, orbspin)), 18.376940955391518, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t2, orbspin)),-77.381111327886615, 8)
        self.assertAlmostEqual(e0, -122.80329415715201, 2)

    def test_ucc_eomee_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r1 = cc.addons.spin2spatial(r1, orbspin)
        r2 = cc.addons.spin2spatial(r2, orbspin)
        vec = eom_uccsd.amplitudes_to_vector_ee(r1,r2)
        vec1 = eom_uccsd.eomee_ccsd_matvec(eom_uccsd.EOMEE(ucc1), vec)
        self.assertAlmostEqual(lib.finger(vec1), -2047.2592663371763, 9)

    def test_ucc_eomee_ccsd_diag(self):
        vec1, vec2 = eom_uccsd.EOMEE(ucc1).get_diag()
        self.assertAlmostEqual(lib.finger(vec1),-49.302724279987167, 9)
        self.assertAlmostEqual(lib.finger(vec2), 367.68014914733976, 9)

    def test_ucc_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEESpinFlip(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = eom_uccsd.eomsf_ccsd_matvec(myeom, vec)
        self.assertAlmostEqual(lib.finger(vec1), 1171.4241926361906, 8)

#    def test_ucc_eomip_matvec(self):
#
#    def test_ucc_eomea_matvec(self):

########################################
# With 4-fold symmetry in integrals
# max_memory = 0
# direct = True
    def test_eomee1(self):
        self.assertAlmostEqual(ucc0.e_corr, -0.2133432430989155, 6)
        e,v = ucc0.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = ucc0.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()

