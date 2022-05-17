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
import numpy
import copy
from pyscf import lib, gto, scf, dft
from pyscf import tdscf
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, molsym, mf_bp86, mf_lda, mcol_lda
    molsym = gto.M(
        atom='''
O     0.   0.       0.
H     0.   -0.757   0.587
H     0.   0.757    0.587''',
        spin=2,
        basis='631g',
        symmetry=True)

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
H     0.   0.    0.
H     0.  -0.7   0.7
H     0.   0.7   0.7'''
    mol.basis = '631g'
    mol.spin = 1
    mol.build()

    mf_lda = dft.GKS(mol).set(xc='lda,').newton().run()
    mcol_lda = None
    if mcfun is not None:
        mcol_lda = dft.GKS(mol).set(xc='lda,', collinear='mcol')
        mcol_lda._numint.spin_samples = 6
        mcol_lda = mcol_lda.run()
    mf_bp86 = dft.GKS(molsym).set(xc='bp86').run()

def tearDownModule():
    global mol, molsym, mf_bp86, mf_lda, mcol_lda
    mol.stdout.close()
    del mol, molsym, mf_bp86, mf_lda, mcol_lda

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
    def test_tddft_lda(self):
        td = mf_lda.TDDFT()
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3] * 27.2114), 3.1188199659345495, 5)

    def test_tda_lda(self):
        td = mf_lda.TDA()
        es = td.kernel(nstates=5)[0]
        a,b = td.get_ab()
        nocc, nvir = a.shape[:2]
        nov = nocc * nvir
        e_ref = numpy.linalg.eigh(a.reshape(nov,nov))[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es[:3] * 27.2114), 3.1822953463662254, 5)

    def test_ab_hf(self):
        mf = scf.GHF(molsym).run()
        self._check_against_ab_ks_complex(mf.TDHF(), -6.310739752194785, 0.4564383749329136)

    def test_col_lda_ab_ks(self):
        self._check_against_ab_ks_real(tdscf.gks.TDDFT(mf_lda), -0.25805697292509255, 0.30518896190461875)

    def test_col_gga_ab_ks(self):
        mf_b3lyp = dft.GKS(mol).set(xc='b3lyp')
        mf_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks_real(mf_b3lyp.TDDFT(), -0.3424518104679129, 0.3416541129007508)

    def test_col_mgga_ab_ks(self):
        mf_m06l = dft.GKS(mol).set(xc='m06l')
        mf_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks_real(tdscf.gks.TDDFT(mf_m06l), -0.19581757687144707, 0.3674283579582636)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_lda_ab_ks(self):
        self._check_against_ab_ks_complex(mcol_lda.TDDFT(), (-1.0869078807096237+0j), (0.3873525612834711+0j))

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gga_ab_ks(self):
        mcol_b3lyp = dft.GKS(mol).set(xc='b3lyp', collinear='mcol')
        mcol_b3lyp._numint.spin_samples = 6
        mcol_b3lyp.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks_complex(mcol_b3lyp.TDDFT(), (-0.4031178158856909+0j), (0.962818119721807+0j))

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_mgga_ab_ks(self):
        mcol_m06l = dft.GKS(mol).set(xc='m06,', collinear='mcol')
        mcol_m06l._numint.spin_samples = 6
        mcol_m06l.__dict__.update(scf.chkfile.load(mf_lda.chkfile, 'scf'))
        self._check_against_ab_ks_complex(mcol_m06l.TDDFT(), (6.920061239623173+0j), (8.756118135792633+0j))

    def _check_against_ab_ks_real(self, td, refa, refb):
        mf = td._scf
        a, b = td.get_ab()
        self.assertAlmostEqual(lib.fp(a), refa, 12)
        self.assertAlmostEqual(lib.fp(b), refb, 12)
        ftda = mf.TDA().gen_vind()[0]
        ftdhf = td.gen_vind()[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 1)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))

        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 12)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b.conj(), x)
        ab2-= numpy.einsum('iajb,jb->ia', a.conj(), y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 12)

    def _check_against_ab_ks_complex(self, td, refa, refb):
        mf = td._scf
        a, b = td.get_ab()
        self.assertAlmostEqual(lib.fp(a), refa, 12)
        self.assertAlmostEqual(lib.fp(b), refb, 12)
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

    def test_tda_with_wfnsym(self):
        td = mf_bp86.TDA()
        td.wfnsym = 'B1'
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.fp(es), 0.4523465502706356, 6)

    def test_tdhf_with_wfnsym(self):
        mf_ghf = scf.GHF(molsym).run()
        td = mf_ghf.TDHF()
        td.wfnsym = 'B1'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.48380638923581476, 6)

    def test_tddft_with_wfnsym(self):
        td = mf_bp86.CasidaTDDFT()
        td.wfnsym = 'B1'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.45050838461527387, 6)


if __name__ == "__main__":
    print("Full Tests for TD-GKS")
    unittest.main()
