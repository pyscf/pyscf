#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

from pyscf import gto, lib
from pyscf import scf
from pyscf import cc
from pyscf.cc import dfuccsd, eom_uccsd

def setUpModule():
    global mol, rhf, mf, ucc, ucc1, nocca, nvira, noccb, nvirb
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol).density_fit(auxbasis='weigend')
    rhf.conv_tol_grad = 1e-8
    rhf.kernel()
    mf = scf.addons.convert_to_uhf(rhf)

    ucc = dfuccsd.UCCSD(mf).run(conv_tol=1e-10)

    mol1 = mol.copy()
    mol1.spin = 2
    mol1.build()
    mf0 = scf.UHF(mol1).density_fit(auxbasis='weigend')
    mf0 = mf0.run(conv_tol=1e-12)
    mf1 = mf0.copy()

    nocca, noccb = mol1.nelec
    nmo = mol1.nao_nr()
    nvira, nvirb = nmo - nocca, nmo-noccb
    numpy.random.seed(12)
    mf1.mo_coeff = numpy.random.random((2, nmo, nmo)) - 0.5
    gmf = scf.addons.convert_to_ghf(mf1)
    orbspin = gmf.mo_coeff.orbspin

    ucc1 = dfuccsd.UCCSD(mf1)

    numpy.random.seed(11)
    no = nocca + noccb
    nv = nvira + nvirb
    r1 = numpy.random.random((no, nv)) - .9
    r2 = numpy.random.random((no, no, nv, nv)) - .9
    r2 = r2 - r2.transpose(1, 0, 2, 3)
    r2 = r2 - r2.transpose(0, 1, 3, 2)
    r1 = cc.addons.spin2spatial(r1, orbspin)
    r2 = cc.addons.spin2spatial(r2, orbspin)
    r1, r2 = eom_uccsd.vector_to_amplitudes_ee(
        eom_uccsd.amplitudes_to_vector_ee(r1, r2), ucc1.nmo, ucc1.nocc)
    ucc1.t1 = r1
    ucc1.t2 = r2

def tearDownModule():
    global mol, rhf, mf, ucc, ucc1
    mol.stdout.close()
    del mol, rhf, mf, ucc, ucc1

class KnownValues(unittest.TestCase):
    def test_with_df(self):
        self.assertAlmostEqual(ucc.e_tot, -76.118403942938741, 6)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.uccsd.UCCSD(mf).ao2mo(mo_coeff)
        self.assertAlmostEqual(lib.fp(eris.oooo),   4.96203346086189, 11)
        self.assertAlmostEqual(lib.fp(eris.ovoo),  -1.36660785172450, 11)
        self.assertAlmostEqual(lib.fp(eris.ovov), 125.80972789115610, 11)
        self.assertAlmostEqual(lib.fp(eris.oovv),  55.12252557132108, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvo), 133.48517302161093, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvv),  59.41874702857587, 11)
        self.assertAlmostEqual(lib.fp(eris.vvvv),  43.56245722797580, 11)
        self.assertAlmostEqual(lib.fp(eris.OOOO),-407.06688974686460, 11)
        self.assertAlmostEqual(lib.fp(eris.OVOO),  56.27143890752203, 11)
        self.assertAlmostEqual(lib.fp(eris.OVOV),-287.70639282707754, 11)
        self.assertAlmostEqual(lib.fp(eris.OOVV), -85.47974577569636, 11)
        self.assertAlmostEqual(lib.fp(eris.OVVO),-228.18507149174869, 11)
        self.assertAlmostEqual(lib.fp(eris.OVVV), -10.71277146732805, 11)
        self.assertAlmostEqual(lib.fp(eris.VVVV), -89.90585731601215, 11)
        self.assertAlmostEqual(lib.fp(eris.ooOO),-336.66684771631481, 11)
        self.assertAlmostEqual(lib.fp(eris.ovOO), -16.41425875389384, 11)
        self.assertAlmostEqual(lib.fp(eris.ovOV), 231.59582076434012, 11)
        self.assertAlmostEqual(lib.fp(eris.ooVV),  20.33912803565480, 11)
        self.assertAlmostEqual(lib.fp(eris.ovVO), 206.47963466976572, 11)
        self.assertAlmostEqual(lib.fp(eris.ovVV), -71.27033893003525, 11)
        self.assertAlmostEqual(lib.fp(eris.vvVV), 172.46980782354174, 11)
        self.assertAlmostEqual(lib.fp(eris.OVoo), -19.93667104163490, 11)
        self.assertAlmostEqual(lib.fp(eris.OOvv), -27.76467892422170, 11)
        self.assertAlmostEqual(lib.fp(eris.OVvo),-140.08585937729470, 11)
        self.assertAlmostEqual(lib.fp(eris.OVvv),  40.69754434429569, 11)

    def test_df_ipccsd(self):
        eom = ucc.eomip_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.42788191076505, 5)
        e,v = ucc.ipccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.42788191078779, 5)
        self.assertAlmostEqual(e[2], 0.50229583182010, 5)
        self.assertAlmostEqual(e[4], 0.68557653039451, 5)

        e,v = ucc.ipccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.50229583182010, 5)

    def test_df_ipccsd_koopmans(self):
        e,v = ucc.ipccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.42788191078779, 5)
        self.assertAlmostEqual(e[2], 0.50229583182010, 5)
        self.assertAlmostEqual(e[4], 0.68557653039451, 5)

    def test_df_eaccsd(self):
        eom = ucc.eomea_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.19038860308234, 5)
        e,v = ucc.eaccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.19038860335165, 5)
        self.assertAlmostEqual(e[2], 0.28339727179550, 5)
        self.assertAlmostEqual(e[4], 0.52224978073482, 5)

        e,v = ucc.eaccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.28339727179551, 5)

    def test_df_eaccsd_koopmans(self):
        e,v = ucc.eaccsd(nroots=6, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19038860335165, 5)
        self.assertAlmostEqual(e[2], 0.28339727179550, 5)

    def test_eomee(self):
        self.assertAlmostEqual(ucc.e_corr, -0.13519304930252, 5)
        eom = ucc.eomee_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        self.assertAlmostEqual(e, 0.28107576231548, 5)

        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.28107576231548, 5)
        self.assertAlmostEqual(e[1], 0.28107576231548, 5)
        self.assertAlmostEqual(e[2], 0.28107576231548, 5)
        self.assertAlmostEqual(e[3], 0.30810935830310, 5)

        e,v = ucc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.30810935830310, 5)

    def test_eomee_ccsd_spin_keep(self):
        e, v = ucc.eomee_ccsd(nroots=2, koopmans=False)
        self.assertAlmostEqual(e[0], 0.28107576231548, 5)
        self.assertAlmostEqual(e[1], 0.30810935830310, 5)

        e, v = ucc.eomee_ccsd(nroots=2, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28107576231548, 5)
        self.assertAlmostEqual(e[1], 0.30810935830310, 5)

    def test_df_eomee_ccsd_matvec(self):
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

        uee1 = eom_uccsd.EOMEESpinKeep(ucc1)
        vec = uee1.amplitudes_to_vector(r1,r2)
        vec1 = uee1.matvec(vec)

        r1, r2 = uee1.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(vec1), 49.6181987143722836, 9)

    def test_df_eomee_ccsd_diag(self):
        vec1, vec2 = eom_uccsd.EOMEE(ucc1).get_diag()
        self.assertAlmostEqual(lib.fp(vec1),  67.5604889872366812, 9)
        self.assertAlmostEqual(lib.fp(vec2), 161.2361154461494834, 9)

    def test_df_eomee_init_guess(self):
        uee = eom_uccsd.EOMEESpinKeep(ucc1)
        diag = uee.get_diag()[0]
        guess = uee.get_init_guess(nroots=1, koopmans=False, diag=diag)
        self.assertAlmostEqual(lib.fp(guess[0]), -0.4558886175539251, 9)

        guess = uee.get_init_guess(nroots=1, koopmans=True, diag=diag)
        self.assertAlmostEqual(lib.fp(guess[0]), -0.8438701329927379, 9)

        guess = uee.get_init_guess(nroots=4, koopmans=False, diag=diag)
        self.assertAlmostEqual(lib.fp(guess), 0.3749885550995292, 9)

        guess = uee.get_init_guess(nroots=4, koopmans=True, diag=diag)
        self.assertAlmostEqual(lib.fp(guess), -0.3812403236695565, 9)

    def test_df_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEESpinFlip(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = myeom.matvec(vec)
        self.assertAlmostEqual(lib.fp(vec1), -1655.6005996668061471, 8)

    def test_ao2mo(self):
        numpy.random.seed(2)
        mo = numpy.random.random(mf.mo_coeff.shape)
        mycc = cc.CCSD(mf).density_fit(auxbasis='ccpvdz-ri')
        mycc.max_memory = 0
        eri_df = mycc.ao2mo(mo)

        self.assertAlmostEqual(lib.fp(eri_df.oooo),-493.98003157749906, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovoo),-203.89515661847452, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovov), -57.62195194777571, 9)
        self.assertAlmostEqual(lib.fp(eri_df.oovv), -91.84858398271636, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvo), -14.88387735916913, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvv), -24.35941895353339, 9)
        self.assertTrue(eri_df.vvvv is None)
        self.assertAlmostEqual(lib.fp(eri_df.OOOO), 144.14457267205376, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVOO),-182.57213907795114, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVOV), 462.82041314370520, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OOVV), 165.48054495591805, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVVO), 499.35368006930844, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVVV), 117.15437910286980, 9)
        self.assertTrue(eri_df.VVVV is None)
        self.assertAlmostEqual(lib.fp(eri_df.ooOO), -13.89000382034967, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovOO),-256.60884544897004, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovOV), -93.37973542764470, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ooVV), -35.14359736260144, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovVO), -80.53112424767198, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovVV),  -1.10606382100687, 9)
        self.assertTrue(eri_df.vvVV is None)
        self.assertAlmostEqual(lib.fp(eri_df.OVoo),-364.17910423297951, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OOvv), -60.94153644542936, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVvo), 355.78017614135643, 9)
        self.assertAlmostEqual(lib.fp(eri_df.OVvv),  57.10096926407320, 9)
        self.assertAlmostEqual(lib.fp(eri_df.vvL),   -0.51651775168057, 9)
        self.assertAlmostEqual(lib.fp(eri_df.VVL),   -5.60414552806429, 9)


if __name__ == "__main__":
    print("Full Tests for DFUCCSD")
    unittest.main()
