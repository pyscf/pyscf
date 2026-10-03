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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df
from pyscf import hessian

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = '6-31g',
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_rhf_hess(self):
        href = scf.RHF(mol).run().Hessian().kernel()
        h1 = scf.RHF(mol).density_fit().run().Hessian().kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 3)

    def test_hess_in_range_coulomb_context(self):
        # omega set on mol has to reach the metric (P|Q) as well as the
        # 3-center integrals (P|uv), as for the gradients in issue #3434
        omega = 0.3
        mf = scf.RHF(mol).density_fit().run()
        hobj = mf.Hessian()
        with mf.with_df.range_coulomb(omega):
            eref = hobj.partial_hess_elec()
            h1ref = numpy.asarray(hobj.make_h1(mf.mo_coeff, mf.mo_occ))
        with mol.with_range_coulomb(omega):
            e1 = hobj.partial_hess_elec()
            h1 = numpy.asarray(hobj.make_h1(mf.mo_coeff, mf.mo_occ))
        self.assertAlmostEqual(abs(eref - e1).max(), 0, 9)
        self.assertAlmostEqual(abs(h1ref - h1).max(), 0, 9)

    def test_rks_lda_hess(self):
        href = mol.RKS.run(xc='lda,vwn').Hessian().kernel()
        df_h = mol.RKS.density_fit().run(xc='lda,vwn').Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_rks_gga_hess(self):
        href = mol.RKS.run(xc='b3lyp').Hessian().kernel()
        df_h = mol.RKS.density_fit().run(xc='b3lyp').Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_rks_mgga_hess(self):
        href = mol.RKS.run(xc='m06').Hessian().kernel()
        df_h = mol.RKS.density_fit().run(xc='m06').Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_rks_rsh_hess(self):
        href = mol.RKS.run(xc='camb3lyp').Hessian().kernel()
        df_h = mol.RKS.density_fit().run(xc='camb3lyp').Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_uhf_hess(self):
        href = scf.UHF(mol).run().Hessian().kernel()
        df_h = scf.UHF(mol).density_fit().run().Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_uks_hess(self):
        href = mol.UKS.run(xc='camb3lyp').Hessian().kernel()
        df_h = mol.UKS.density_fit().run(xc='camb3lyp').Hessian()
        df_h.auxbasis_response = 2
        h1 = df_h.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

    def test_uks_lda_hess(self):
        href = mol.UKS.run(xc='svwn').Hessian().kernel()
        mf = mol.UKS(xc='svwn').density_fit().run()
        df_hess = mf.Hessian()
        df_hess.auxbasis_response = 2
        h1 = df_hess.kernel()
        self.assertAlmostEqual(abs(href - h1).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for df.hessian")
    unittest.main()
