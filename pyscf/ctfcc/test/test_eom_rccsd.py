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


import ctf
import numpy
import unittest
from pyscf import gto, scf, cc
from pyscf.ctfcc import rccsd, eom_rccsd
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
mycc.eris = mycc.ao2mo()
refcc.eris = refcc.ao2mo()

e, t1, t2 = mycc.init_amps(eris=mycc.eris)
e1, t1r, t2r = refcc.init_amps(eris=refcc.eris)
mycc.t1, mycc.t2 = t1, t2
refcc.t1, refcc.t2 = t1r, t2r

myip = eom_rccsd.EOMIP(mycc)
myea = eom_rccsd.EOMEA(mycc)

refip = cc.eom_rccsd.EOMIP(refcc)
refea = cc.eom_rccsd.EOMEA(refcc)

imds = myip.make_imds()
imds.make_ea()

imdsr = refip.make_imds()
imdsr.make_ea()

class RefValues(unittest.TestCase):
    def test_imds(self):
        for key in imds.__dict__.keys():
            if key[0] in ['t', 'L','F', "W"]:
                x = getattr(imds, key).array.to_nparray()
                y = getattr(imdsr,key)
                self.assertTrue(numpy.linalg.norm(x - y)<1e-10)

    def test_matvec(self):
        vec = ctf.random.random([myip.vector_size()])
        vecnew = myip.matvec(vec, imds=imds)
        vecr = vec.to_nparray()
        vecnewr = refip.matvec(vecr, imds=imdsr)
        self.assertTrue(numpy.linalg.norm(vec.to_nparray()-vecr)<1e-10)
        self.assertTrue(numpy.linalg.norm(vecnew.to_nparray()-vecnewr)<1e-10)

        vec = ctf.random.random([myea.vector_size()])
        vecnew = myea.matvec(vec, imds=imds)
        vecr = vec.to_nparray()
        vecnewr = refea.matvec(vecr, imds=imdsr)
        self.assertTrue(numpy.linalg.norm(vec.to_nparray()-vecr)<1e-10)
        self.assertTrue(numpy.linalg.norm(vecnew.to_nparray()-vecnewr)<1e-10)

if __name__ == '__main__':
    print("eom_rccsd tests")
    unittest.main()
