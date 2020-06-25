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
#
# Author: Yang Gao <younggao1994@gmail.com>


import numpy
import unittest
from pyscf import gto, scf, cc
from pyscf.ctfcc import rccsd
from symtensor.sym_ctf import tensor


mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.max_cycle = 1
mf.kernel()

mycc = rccsd.RCCSD(mf)
refcc = cc.RCCSD(mf)
eris = mycc.ao2mo()
erisr = refcc.ao2mo()
class REFTEST(unittest.TestCase):
    def test_eris(self):
        for key in ['oooo', 'ovov', 'oovv', 'ovvo']:
            x = getattr(eris, key).array.to_nparray()
            y = getattr(erisr,key)
            self.assertTrue(numpy.linalg.norm(x - y)<1e-10)
        self.assertTrue(numpy.linalg.norm(eris.ooov.array.to_nparray() - erisr.ovoo.transpose(2,3,0,1))<1e-10)

    def test_update_amps(self):
        e1, t1, t2 = mycc.init_amps(eris=eris)
        e1r, t1r, t2r = refcc.init_amps(eris=erisr)
        self.assertTrue(abs(e1-e1r)<1e-10)
        self.assertTrue(numpy.linalg.norm(t1.array.to_nparray()-t1r)<1e-10)
        self.assertTrue(numpy.linalg.norm(t2.array.to_nparray()-t2r)<1e-10)
        t1new, t2new = mycc.update_amps(t1, t2, eris=eris)
        t1newr, t2newr = refcc.update_amps(t1r, t2r, eris=erisr)
        self.assertTrue(numpy.linalg.norm(t1new.array.to_nparray()-t1newr)<1e-10)
        self.assertTrue(numpy.linalg.norm(t2new.array.to_nparray()-t2newr)<1e-10)


if __name__ == '__main__':
    print("rccsd tests")
    unittest.main()
