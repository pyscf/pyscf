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
from pyscf import lib, gto, scf, dft
from pyscf import tdscf
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
    mol.basis = 'uncsto3g'
    mol.spin = 1
    mol.build()

    mf_lda = mol.DKS().set(xc='lda,', conv_tol=1e-12,
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
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_tddft_lda(self):
        td = mf_lda.TDDFT()
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 8)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3] * 27.2114), 3.157449136045103, 5)

    def test_tda_lda(self):
        td = mf_lda.TDA()
        es = td.kernel(nstates=5)[0]
        a,b = td.get_ab()
        nocc, nvir = a.shape[:2]
        nov = nocc * nvir
        e_ref = numpy.linalg.eigh(a.reshape(nov,nov))[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3] * 27.2114), 3.220469947746697, 5)

    def test_ab_hf(self):
        mf = scf.DHF(mol).run()
        self._check_against_ab_ks(mf.TDHF(), 3.162381949829394, 0.11755137875158098)

    def test_col_lda_ab_ks(self):
        self._check_against_ab_ks(mf_lda.TDDFT(), 2.6168030250127075, 0.10466808380307921)

    def test_col_gga_ab_ks(self):
        mf_b3lyp = dft.DKS(mol).set(xc='b3lyp5')
        mf_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mf_b3lyp.TDDFT(), 2.568259113326634, 0.1531691970098629)

    def test_col_mgga_ab_ks(self):
        mf_m06l = dft.DKS(mol).set(xc='m06l')
        mf_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mf_m06l.TDDFT(), 3.382341929143924, 0.6409502830461241)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_lda_ab_ks(self):
        mcol_lda = dft.UDKS(mol).set(xc='lda,', collinear='mcol')
        mcol_lda._numint.spin_samples = 6
        mcol_lda.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_lda.TDDFT(), 2.880950836922392, 0.45543578075726204)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gga_ab_ks(self):
        mcol_b3lyp = dft.UDKS(mol).set(xc='b3lyp5', collinear='mcol')
        mcol_b3lyp._numint.spin_samples = 6
        mcol_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_b3lyp.TDDFT(), 2.753430274014454, 0.607433969753113)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_mgga_ab_ks(self):
        mcol_m06l = dft.UDKS(mol).set(xc='m06l', collinear='mcol')
        mcol_m06l._numint.spin_samples = 6
        mcol_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks(mcol_m06l.TDDFT())

    def _check_against_ab_ks(self, td, refa=None, refb=None):
        mf = td._scf
        a, b = td.get_ab()
        if refa is not None:
            self.assertAlmostEqual(lib.fp(abs(a)), refa, 4)
        if refb is not None:
            self.assertAlmostEqual(lib.fp(abs(b)), refb, 4)
        ftda = mf.TDA().gen_vind()[0]
        ftdhf = td.gen_vind()[0]
        n2c = mf.mo_occ.size // 2
        nocc = numpy.count_nonzero(mf.mo_occ == 1)
        nvir = numpy.count_nonzero(mf.mo_occ == 0) - n2c
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


if __name__ == "__main__":
    print("Full Tests for TD-DKS")
    unittest.main()
