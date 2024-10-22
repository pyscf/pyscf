#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy
from pyscf import lib, gto, scf
from pyscf.dft import radi
from pyscf.x2c import x2c, dft, tdscf
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, mf_lda
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
    H     0.   0.    0.
    H     0.  -0.7   0.7
    H     0.   0.7   0.7'''
    mol.basis = '6-31g'
    mol.spin = 1
    mol.build()

    mf_lda = dft.UKS(mol).set(xc='lda,', conv_tol=1e-12,
                              chkfile=tempfile.NamedTemporaryFile().name).newton().run()

def tearDownModule():
    global mol, mf_lda
    mol.stdout.close()
    del mol, mf_lda

def diagonalize(a, b, nroots=4):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = numpy.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e = numpy.linalg.eig(numpy.asarray(h))[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_tddft_lda(self):
        td = mf_lda.TDDFT()
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 8)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 3.119041718921026, 5)

    def test_tda_lda(self):
        td = mf_lda.TDA()
        es = td.kernel(nstates=5)[0]
        a,b = td.get_ab()
        nocc, nvir = a.shape[:2]
        nov = nocc * nvir
        e_ref = numpy.linalg.eigh(a.reshape(nov,nov))[0]
        es = td.kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3] * 27.2114), 3.1825211067032253, 5)

    def test_ab_hf(self):
        mf = x2c.UHF(mol).newton().run(conv_tol=1e-12)
        self._check_against_ab_ks(mf.TDHF(), -0.2404548371794495, 0.6508765417771681, 4)

    def test_col_lda_ab_ks(self):
        self._check_against_ab_ks(mf_lda.TDDFT(), -0.5231134770778959, 0.07879428138412828)

    def test_col_gga_ab_ks(self):
        mf_b3lyp = dft.UKS(mol).set(xc='b3lyp5')
        mf_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mf_b3lyp.TDDFT(), -0.4758219953792988, 0.17715631269859033)

    def test_col_mgga_ab_ks(self):
        mf_m06l = dft.UKS(mol).run(xc='m06l', conv_tol=1e-12)
        mf_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mf_m06l.TDDFT(), -0.4919270127924622, 0.14597029880651433, places=5)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_lda_ab_ks(self):
        mcol_lda = dft.UKS(mol).set(xc='lda,', collinear='mcol')
        mcol_lda._numint.spin_samples = 6
        mcol_lda.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_lda.TDDFT(), -0.6154532929747091, 0.49991930461632084, places=5)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gga_ab_ks(self):
        mcol_b3lyp = dft.UKS(mol).set(xc='b3lyp5', collinear='mcol')
        mcol_b3lyp._numint.spin_samples = 6
        mcol_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_b3lyp.TDDFT(), -0.4954910129906521, 0.4808365159189027)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_mgga_ab_ks(self):
        mcol_m06l = dft.UKS(mol).set(xc='m06l', collinear='mcol')
        mcol_m06l._numint.spin_samples = 6
        mcol_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_m06l.TDDFT(), -0.6984240332038076, 2.0192987108288794, 5)

    def _check_against_ab_ks(self, td, refa, refb, places=6):
        mf = td._scf
        a, b = td.get_ab()
        self.assertAlmostEqual(lib.fp(abs(a)), refa, places)
        self.assertAlmostEqual(lib.fp(abs(b)), refb, places)
        ftda = mf.TDA().gen_vind()[0]
        ftdhf = td.gen_vind()[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 1)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = (numpy.random.random((2,nocc,nvir)) +
                     numpy.random.random((2,nocc,nvir)) * 1j)

        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 12)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b.conj(), x)
        ab2-= numpy.einsum('iajb,jb->ia', a.conj(), y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 12)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_vs_gks(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=20):
            mol = gto.M(atom='C', basis='6-31g')
            ref = dft.UKS(mol)
            ref.xc = 'pbe'
            ref.collinear = 'mcol'
            ref._numint.spin_samples = 6
            ref.run()
            td = ref.TDA()
            td.positive_eig_threshold = -10
            eref = td.kernel(nstates=5)[0]

            c = numpy.vstack(mol.sph2spinor_coeff())
            mo1 = c.dot(ref.mo_coeff)
            dm = ref.make_rdm1(mo1, ref.mo_occ)
            mf = mol.GKS().x2c1e()
            mf.xc = 'pbe'
            mf.collinear = 'mcol'
            mf._numint.spin_samples = 6
            mf.max_cycle = 1
            mf.kernel(dm0=dm)
            td = mf.TDA()
            td.positive_eig_threshold = -10
            es = td.kernel(nstates=5)[0]
            self.assertAlmostEqual(abs(es - eref).max(), 0, 6)


if __name__ == "__main__":
    print("Full Tests for TD-X2C-KS")
    unittest.main()
