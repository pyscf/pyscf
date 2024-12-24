#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
#

import unittest
import numpy as np
from pyscf import __config__
from pyscf.pbc import gto, scf, tdscf, cc
from pyscf import gto as molgto, scf as molscf, tdscf as moltdscf
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEESinglet
from pyscf.data.nist import HARTREE2EV as unitev

def diagonalize(a, b, nroots=4):
    nkpts, nocc, nvir = a.shape[:3]
    a = a.reshape(nkpts*nocc*nvir, -1)
    b = b.reshape(nkpts*nocc*nvir, -1)
    h = np.block([[a        , b       ],
                  [-b.conj(),-a.conj()]])
    e = np.linalg.eigvals(np.asarray(h))
    lowest_e = np.sort(e[e.real > 0].real)
    lowest_e = lowest_e[lowest_e > 1e-3][:nroots]
    return lowest_e

class Diamond(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
        cell.a = '''
        1.7850000000 1.7850000000 0.0000000000
        0.0000000000 1.7850000000 1.7850000000
        1.7850000000 0.0000000000 1.7850000000
        '''
        cell.pseudo = 'gth-hf-rev'
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.precision = 1e-10
        cell.build()
        kpts = cell.make_kpts((2,1,1))
        mf = scf.KRHF(cell, kpts=kpts).rs_density_fit(auxbasis='weigend').run()
        cls.cell = cell
        cls.mf = mf

        cls.nstates = 4 # make sure first `nstates_test` states are converged
        cls.nstates_test = 1
    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.mf

    def kernel(self, TD, ref, kshift_lst, **kwargs):
        td = TD(self.mf).set(kshift_lst=kshift_lst,
                             nstates=self.nstates, **kwargs).run()
        for kshift,e in enumerate(td.e):
            n = len(ref[kshift])
            self.assertAlmostEqual(abs(e[:n] * unitev - ref[kshift]).max(), 0, 4)
        return td

    def test_tda_singlet_eomccs(self):
        ''' Brute-force solution to the KTDA equation. Compared to the brute-force
            implementation of KTDA from EOM-EE-CCSD.
        '''
        td = tdscf.KTDA(self.mf).set(kshift_lst=np.arange(len(self.mf.kpts)))
        ecis_k = []
        for kshift in td.kshift_lst:
            vind, hdiag = td.gen_vind(td._scf, kshift)
            heff = vind(np.eye(hdiag.size))
            ecis_k.append(np.linalg.eigh(heff)[0])

        mcc = cc.KCCSD(self.mf)
        mcc.kernel()
        meom = EOMEESinglet(mcc)
        ecc_k = meom.cis(nroots=ecis_k[0].size, kptlist=td. kshift_lst)[0]

        for e,ecc in zip(ecis_k, ecc_k):
            self.assertAlmostEqual(abs(e * unitev  - ecc.real * unitev).max(), 0, 3)

    def test_tda_singlet(self):
        ref = [[10.9573977036],
               [11.0418311085]]
        td = self.kernel(tdscf.KTDA, ref, np.arange(2))
        a0, _ = td.get_ab(kshift=0)
        nk, no, nv = a0.shape[:3]
        a0 = a0.reshape(nk*no*nv,-1)
        eref0 = np.linalg.eigvalsh(a0)[:4]
        a1, _ = td.get_ab(kshift=1)
        nk, no, nv = a1.shape[:3]
        a1 = a1.reshape(nk*no*nv,-1)
        eref1 = np.linalg.eigvalsh(a1)[:4]
        self.assertAlmostEqual(abs(td.e[0][:2] - eref0[:2]).max(), 0, 5)
        self.assertAlmostEqual(abs(td.e[1][:2] - eref1[:2]).max(), 0, 5)

        vind, hdiag = td.gen_vind(td._scf, kshift=0)
        z = a0[:1]
        self.assertAlmostEqual(abs(vind(z) - a0.dot(z[0])).max(), 0, 10)
        vind, hdiag = td.gen_vind(td._scf, kshift=1)
        self.assertAlmostEqual(abs(vind(z) - a1.dot(z[0])).max(), 0, 10)

    def test_tda_triplet(self):
        ref = [[6.4440137833],
               [7.4264899075]]
        self.kernel(tdscf.KTDA, ref, np.arange(2), singlet=False)

    def test_tdhf_singlet(self):
        # The first state can be obtained by solving nstates=9
        #ref = [[10.60761946, 10.76654619, 10.76654619]]
        ref = [[10.76654619, 10.76654619]]
        td = self.kernel(tdscf.KTDHF, ref, [0])
        a0, b0 = td.get_ab(kshift=0)
        eref0 = diagonalize(a0, b0)
        self.assertAlmostEqual(abs(td.e[0][:2] - eref0[1:3]).max(), 0, 5)

        nk, no, nv = a0.shape[:3]
        nkov = nk * no * nv
        a0 = a0.reshape(nkov,-1)
        b0 = b0.reshape(nkov,-1)
        z = np.hstack([a0[:1], a0[1:2]])

        h = np.block([[ a0       , b0       ],
                      [-b0.conj(),-a0.conj()]])
        vind, hdiag = td.gen_vind(td._scf, kshift=0)
        self.assertAlmostEqual(abs(vind(z) - h.dot(z[0])).max(), 0, 10)

    def test_tdhf_triplet(self):
        ref = [[5.9794378466]]
        self.kernel(tdscf.KTDHF, ref, [0], singlet=False)


class WaterBigBox(unittest.TestCase):
    ''' Match molecular CIS
    '''
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()
        kpts = cell.make_kpts((2,1,1))
        mf = scf.KRHF(cell, kpts=kpts).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = cell.to_mol()
        molmf = molscf.RHF(mol).density_fit(auxbasis=mf.with_df.auxbasis).run()
        cls.mol = mol
        cls.molmf = molmf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 3

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, MOLTD, **kwargs):
        td = TD(self.mf).set(nstates=self.nstates, **kwargs).run()
        moltd = MOLTD(self.molmf).set(nstates=self.nstates, **kwargs).run()
        ref = moltd.e[:self.nstates_test]
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev  - ref * unitev).max(), 0, 2)

    def test_tda_singlet(self):
        self.kernel(tdscf.KTDA, moltdscf.TDA)

    def test_tda_triplet(self):
        self.kernel(tdscf.KTDA, moltdscf.TDA, singlet=False)

    def test_tdhf_singlet(self):
        self.kernel(tdscf.KTDHF, moltdscf.TDHF)

    def test_tdhf_triplet(self):
        self.kernel(tdscf.KTDHF, moltdscf.TDHF, singlet=False)


if __name__ == "__main__":
    print("Full Tests for krhf-TDA and krhf-TDHF")
    unittest.main()
