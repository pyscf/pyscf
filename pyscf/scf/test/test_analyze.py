#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf.scf import analyze

class KnownValues(unittest.TestCase):
    def test_bond_order(self):
        h2 = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
        mf_h2 = scf.RHF(h2).run()
        self.assertAlmostEqual(mf_h2.mayer_bond_order(verbose=0)[0,1], 1, 9)
        self.assertAlmostEqual(mf_h2.wiberg_bond_order(verbose=0)[0,1], 1, 9)

        c2h2 = gto.M(atom='C 0 0 .601; C 0 0 -.601; H 0 0 1.663; H 0 0 -1.663',
                     basis='sto-3g', verbose=0)
        mf_c2h2 = scf.RHF(c2h2).run()
        self.assertAlmostEqual(mf_c2h2.mayer_bond_order(verbose=0)[0,1], 2.99737, 4)
        self.assertAlmostEqual(mf_c2h2.wiberg_bond_order(verbose=0)[0,1], 3.00313, 4)

        # Each spin density of a determinant is idempotent, so the bond orders
        # of an atom summed over all atoms, itself included, are twice its
        # Mulliken population (Mayer) or Lowdin population (Wiberg).
        ion = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                    basis='6-31g*', charge=1, spin=1, verbose=0)
        nao = ion.nao
        s = ion.intor('int1e_ovlp')
        e, u = numpy.linalg.eigh(s)
        s_half = numpy.dot(u * e**.5, u.T)
        for mf1 in (scf.UHF(ion).run(), scf.ROHF(ion).run(), scf.GHF(ion).run()):
            dm = numpy.asarray(mf1.make_rdm1())
            if dm.ndim == 2:
                dm = dm[:nao,:nao] + dm[nao:,nao:]
            else:
                dm = dm[0] + dm[1]
            pops = [numpy.einsum('ij,ji->i', dm, s),
                    numpy.diag(s_half.dot(dm).dot(s_half))]
            for method, pop in zip(('mayer', 'wiberg'), pops):
                n_atom = [pop[p0:p1].sum() for p0, p1 in ion.aoslice_by_atom()[:,2:]]
                bo = analyze._bond_order_matrix(ion, mf1.make_rdm1(), mf1.get_ovlp(), method)
                self.assertAlmostEqual(abs(bo.sum(axis=1) - numpy.array(n_atom)*2).max(), 0, 9)

        # RHF, UHF and GHF agree for a closed-shell molecule
        h2o = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                    basis='6-31g*', verbose=0)
        ref = scf.RHF(h2o).run(conv_tol=1e-12)
        for mf1 in (scf.UHF(h2o), scf.GHF(h2o)):
            mf1.run(conv_tol=1e-12)
            self.assertAlmostEqual(abs(mf1.mayer_bond_order(verbose=0) -
                                       ref.mayer_bond_order(verbose=0)).max(), 0, 6)
            self.assertAlmostEqual(abs(mf1.wiberg_bond_order(verbose=0) -
                                       ref.wiberg_bond_order(verbose=0)).max(), 0, 6)


if __name__ == "__main__":
    print("Tests for scf.analyze")
    unittest.main()
