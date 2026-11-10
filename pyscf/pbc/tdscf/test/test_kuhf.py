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
from pyscf import lib
from pyscf.pbc import gto, scf, dft, tdscf
from pyscf.pbc.tdscf import kuhf
from pyscf import gto as molgto, scf as molscf, tdscf as moltdscf
from pyscf.data.nist import HARTREE2EV as unitev


def diagonalize(a, b, nroots=4):
    a = spin_orbital_block(a)
    b = spin_orbital_block(b, True)
    abba = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e = np.linalg.eig(abba)[0]
    lowest_e = np.sort(e[e.real > 0].real)
    lowest_e = lowest_e[lowest_e > 1e-3][:nroots]
    return lowest_e

def spin_orbital_block(a, symmetric=False):
    a_aa, a_ab, a_bb = a
    nkpts, nocc_a, nvir_a, _, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nkpts*nocc_a*nvir_a,nkpts*nocc_a*nvir_a))
    a_ab = a_ab.reshape((nkpts*nocc_a*nvir_a,nkpts*nocc_b*nvir_b))
    if symmetric:
        a_ba = a_ab.T
    else:
        a_ba = a_ab.conj().T
    a_bb = a_bb.reshape((nkpts*nocc_b*nvir_b,nkpts*nocc_b*nvir_b))
    a = np.block([[a_aa, a_ab],
                  [a_ba, a_bb]])
    return a

#class Diamond(unittest.TestCase):
#    ''' Reproduce KRHF-TDSCF
#    '''
#    @classmethod
#    def setUpClass(cls):
#        cell = gto.Cell()
#        cell.verbose = 4
#        cell.output = '/dev/null'
#        cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
#        cell.a = '''
#        1.7850000000 1.7850000000 0.0000000000
#        0.0000000000 1.7850000000 1.7850000000
#        1.7850000000 0.0000000000 1.7850000000
#        '''
#        cell.pseudo = 'gth-hf-rev'
#        cell.basis = {'C': [[0, (0.8, 1.0)],
#                            [1, (1.0, 1.0)]]}
#        cell.precision = 1e-10
#        cell.build()
#        kpts = cell.make_kpts((2,1,1))
#        mf = scf.KUHF(cell, kpts=kpts).rs_density_fit(auxbasis='weigend').run()
#        cls.cell = cell
#        cls.mf = mf
#
#        cls.nstates = 5 # make sure first `nstates_test` states are converged
#        cls.nstates_test = 2
#    @classmethod
#    def tearDownClass(cls):
#        cls.cell.stdout.close()
#        del cls.cell, cls.mf
#
#    def kernel(self, TD, ref, kshift_lst, **kwargs):
#        td = TD(self.mf).set(kshift_lst=kshift_lst,
#                             nstates=self.nstates, **kwargs).run()
#        for kshift,e in enumerate(td.e):
#            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev  - ref[kshift]).max(), 0, 4)
#        return td
#
#    def test_tda(self):
#        # same as lowest roots in Diamond->test_tda_singlet/triplet in test_krhf.py
#        ref = [[6.4440137833, 7.5317890777],
#               [7.4264899075, 7.6381352853]]
#        td = self.kernel(tdscf.KTDA, ref, np.arange(len(ref)), conv_tol=1e-7)
#        a0, _ = td.get_ab(kshift=0)
#        a0 = spin_orbital_block(a0)
#        eref0 = np.linalg.eigvalsh(a0)[:4]
#        a1, _ = td.get_ab(kshift=1)
#        a1 = spin_orbital_block(a1)
#        eref1 = np.linalg.eigvalsh(a1)[:4]
#        self.assertAlmostEqual(abs(td.e[0][:2] - eref0[:2]).max(), 0, 5)
#        self.assertAlmostEqual(abs(td.e[1][:2] - eref1[:2]).max(), 0, 5)
#
#        vind, hdiag = td.gen_vind(td._scf, kshift=0)
#        z = a0[:1]
#        self.assertAlmostEqual(abs(vind(z) - a0.dot(z[0])).max(), 0, 10)
#        vind, hdiag = td.gen_vind(td._scf, kshift=1)
#        self.assertAlmostEqual(abs(vind(z) - a1.dot(z[0])).max(), 0, 10)
#
#    def test_tdhf(self):
#        # same as lowest roots in Diamond->test_tdhf_singlet/triplet in test_krhf.py
#        ref = [[5.9794378466, 5.9794378466]]
#        td = self.kernel(tdscf.KTDHF, ref, [0])
#
#        a0, b0 = td.get_ab(kshift=0)
#        eref0 = diagonalize(a0, b0)
#        self.assertAlmostEqual(abs(td.e[0][:4] - eref0[:4]).max(), 0, 5)
#
#        a0 = spin_orbital_block(a0)
#        b0 = spin_orbital_block(b0, True)
#        h = np.block([[ a0       , b0       ],
#                      [-b0.conj(),-a0.conj()]])
#        z = np.hstack([a0[:1], a0[1:2]])
#        vind, hdiag = td.gen_vind(td._scf, kshift=0)
#        self.assertAlmostEqual(abs(vind(z) - h.dot(z[0])).max(), 0, 10)


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
        cell.spin = 4   # TOTAL spin in the corresponding supercell
        cell.a = np.eye(3) * 15
        cell.basis = 'sto-3g'
        cell.build()
        kpts = cell.make_kpts((2,1,1))
        mf = scf.KUHF(cell, kpts=kpts).rs_density_fit(auxbasis='weigend')
        mf.with_df.omega = 0.1
        mf.kernel()
        cls.cell = cell
        cls.mf = mf

        mol = molgto.Mole()
        for key in ['verbose','output','atom','basis']:
            setattr(mol, key, getattr(cell, key))
        mol.spin = cell.spin // len(kpts)
        mol.build()
        molmf = molscf.UHF(mol).density_fit(auxbasis=mf.with_df.auxbasis).run()
        cls.mol = mol
        cls.molmf = molmf

        cls.nstates = 5 # make sure first `nstates_test` states are converged
        cls.nstates_test = 2

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.mol.stdout.close()
        del cls.cell, cls.mf
        del cls.mol, cls.molmf

    def kernel(self, TD, MOLTD, **kwargs):
        td = TD(self.mf).set(nstates=self.nstates, **kwargs).run()
        moltd = MOLTD(self.molmf).set(nstates=self.nstates, **kwargs).run()
        ref = moltd.e
        for kshift,e in enumerate(td.e):
            self.assertAlmostEqual(abs(e[:self.nstates_test] * unitev -
                                       ref[:self.nstates_test] * unitev).max(), 0, 2)

    def test_tda(self):
        self.kernel(tdscf.KTDA, moltdscf.TDA)

    def test_tdhf(self):
        self.kernel(tdscf.KTDHF, moltdscf.TDHF)


class SmallCellResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.M(atom='Be 0 0 0', a=np.eye(3)*3,
                     basis=[[0, [.8, 1]], [0, [.3, 1]], [1, [.5, 1]]],
                     mesh=[15]*3, precision=1e-8, verbose=0)
        cls.mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
        cls.mf = cls.mf.density_fit(auxbasis='weigend').run(conv_tol=1e-11).to_uhf()

    def check_ab(self, xc=None):
        mf = self.mf
        if xc is not None:
            mf = mf.to_ks(xc=xc)
            mf.grids = dft.gen_grid.UniformGrids(mf.cell).build()
            # Shift the quadrature away from nuclear zero-gradient points.
            mf.grids.coords += np.sum(mf.cell.lattice_vectors(), axis=0) / 30
        a, b = kuhf.get_ab(mf)
        def blocks(tensors):
            aa, ab, bb = [x.reshape(6, 6) for x in tensors]
            return np.block([[aa, ab], [ab.conj().T, bb]])
        a, b = blocks(a), blocks(b)
        tda = kuhf.TDA(mf)
        vind, diag = tda.gen_vind(mf)
        np.testing.assert_allclose(vind(np.eye(len(diag))).T, a, atol=1e-8)
        td = kuhf.TDHF(mf)
        vind, diag = td.gen_vind(mf)
        full = np.block([[a, b], [-b.conj(), -a.conj()]])
        np.testing.assert_allclose(vind(np.eye(len(diag))).T, full, atol=1e-8)

    def test_hf(self):
        self.check_ab()

    def test_lda(self):
        self.check_ab('lda,vwn')

    def test_gga(self):
        self.check_ab('pbe')

    def test_mgga(self):
        self.check_ab('tpss')



if __name__ == "__main__":
    print("Full Tests for kuhf-TDA and kuhf-TDHF")
    unittest.main()
