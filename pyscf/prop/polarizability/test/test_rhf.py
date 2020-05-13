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
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import tdscf
from pyscf.prop.polarizability import rhf

class KnowValues(unittest.TestCase):
    def test_polarizability_with_freq(self):
        mol = gto.M(atom='''O      0.   0.       0.
                            H      0.  -0.757    0.587
                            H      0.   0.757    0.587''',
                    basis='6-31g')
        mf = scf.RHF(mol).run(conv_tol=1e-14)

        mo_coeff = mf.mo_coeff
        occidx = mf.mo_occ > 0
        orbo = mo_coeff[:, occidx]
        orbv = mo_coeff[:,~occidx]
        nocc = orbo.shape[1]
        nvir = orbv.shape[1]
        nov = nocc*nvir

        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
        with mol.with_common_orig(charge_center):
            int_r = mol.intor_symmetric('int1e_r', comp=3)
        h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
        v1 = h1[:,nocc:].transpose(2,1,0).reshape(nov,-1)
        v = numpy.vstack((v1,v1))

        a, b = tdscf.rhf.get_ab(mf)
        a = a.reshape(nov,nov)
        b = b.reshape(nov,nov)
        mat = numpy.bmat(((a,b),(b,a)))
        freq = 0.1
        mat[:nov,:nov] -= numpy.eye(nov)*freq
        mat[nov:,nov:] += numpy.eye(nov)*freq

        # frequency-dependent property
        u1 = numpy.linalg.solve(mat, v)
        ref = numpy.einsum('px,py->xy', v, u1)*2
        val = rhf.Polarizability(mf).polarizability_with_freq(freq)
        # errors mainly due to krylov solver
        self.assertAlmostEqual(abs(ref-val).max(), 0, 7)

        # static property
        ref = numpy.einsum('px,py->xy', v1, numpy.linalg.solve(a+b, v1))*4
        val = rhf.Polarizability(mf).polarizability()
        self.assertAlmostEqual(abs(ref-val).max(), 0, 7)

        val = rhf.Polarizability(mf).polarizability_with_freq(freq=0)
        # errors mainly due to krylov solver
        self.assertAlmostEqual(abs(ref-val).max(), 0, 7)


    def test_polarizability_with_freq_against_tdhf(self):
        # modified based on https://github.com/pyscf/pyscf/issues/507
        mol = gto.M(atom='''
                O  0.0  0.0  0.0
                O  0.0  0.0  1.5
                H  1.0  0.0  0.0
                H  0.0  0.7  1.0''')
        nmo, nocc = mol.nao, mol.nelec[0]
        nvir = nmo - nocc
        mf = scf.RHF(mol).run()
        mf_td = tdscf.TDHF(mf)
        mf_td.nstates = nocc * nvir
        mf_td.run()
        td_transdip = mf_td.transition_dipole()
        td_eig = mf_td.e

        freq_to_res = lambda omega: numpy.einsum("nt, n, ns -> ts", td_transdip, 1 / (td_eig - omega), td_transdip)
        freq_to_nonres = lambda omega: numpy.einsum("nt, n, ns -> ts", td_transdip, 1 / (td_eig + omega), td_transdip)
        freq_to_alpha = lambda omega: freq_to_res(omega) + freq_to_nonres(omega)

        ref = freq_to_alpha( 0. )
        val = rhf.Polarizability(mf).polarizability_with_freq(freq= 0. )
        self.assertAlmostEqual(abs(ref - val).max(), 0, 8)
        ref = freq_to_alpha( 0.1)
        val = rhf.Polarizability(mf).polarizability_with_freq(freq= 0.1)
        self.assertAlmostEqual(abs(ref - val).max(), 0, 8)

if __name__ == "__main__":
    print("Tests for polarizability")
    unittest.main()

