#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from functools import reduce
import numpy as np
from pyscf.pbc import gto, scf, mp

def setUpModule():
    global cell, kmf, kpts, nkpts
    cell = gto.Cell()
    cell.atom = """
    H  0.0 0.0 0.0
    F  0.9 0.0 0.0
    """
    cell.basis = 'sto-3g'
    cell.a = [[2.82, 0, 0], [0, 2.82, 0], [0, 0, 2.82]]
    cell.dimension = 1
    cell.low_dim_ft_type = 'inf_vacuum'
    cell.output = '/dev/null'
    cell.build()

    nk = [2,1,1]
    kpts = cell.make_kpts(nk)
    nkpts = len(kpts)
    kmf = scf.KRHF(cell, kpts = kpts, exxdiv=None).density_fit()
    kmf.kernel()

def tearDownModule():
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_kmp2_contract_eri_dm(self):
        kmp2 = mp.KMP2(kmf)
        kmp2.kernel()
        e_tot = kmp2.e_tot

        mo_coeff = kmf.mo_coeff
        hcore = kmf.get_hcore()
        for k in range(nkpts):
            hcore[k] = reduce(np.dot, (mo_coeff[k].T.conj(), hcore[k], mo_coeff[k]))

        dm1 = kmp2.make_rdm1()
        dm2 = kmp2.make_rdm2()
        e = 0
        for k in range(nkpts):
            e += np.einsum('pq,qp', dm1[k], hcore[k]).real / nkpts
        ao2mo = kmp2._scf.with_df.ao2mo
        idx = 0
        for kp in range(nkpts):
            for kq in range(nkpts):
                for kr in range(nkpts):
                    ks = kmp2.khelper.kconserv[kp,kq,kr]
                    mop = mo_coeff[kp]
                    moq = mo_coeff[kq]
                    mor = mo_coeff[kr]
                    mos = mo_coeff[ks]
                    eri = ao2mo((mop,moq,mor,mos), 
                          (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                          compact=False).reshape(mop.shape[-1],moq.shape[-1],
                          mor.shape[-1],mos.shape[-1]) / nkpts
                    e += np.einsum('pqrs,pqrs',dm2[idx], eri).real * 0.5 / nkpts
                    idx += 1
        e += cell.energy_nuc()
        self.assertAlmostEqual(e, e_tot, 4)

if __name__ == "__main__":
    print("Full Tests for kmp2 rdm")
    unittest.main()
