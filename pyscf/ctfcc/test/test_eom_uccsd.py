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
from pyscf.ctfcc import uccsd, eom_uccsd
from symtensor.sym_ctf import tensor


mol = gto.Mole()
mol.verbose = 7
mol.spin = 2
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()

mf = scf.UHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = uccsd.UCCSD(mf)
refcc = cc.UCCSD(mf)
mycc.eris = mycc.ao2mo()
refcc.eris = refcc.ao2mo()

e, t1, t2 = mycc.init_amps(eris=mycc.eris)
e1, t1r, t2r = refcc.init_amps(eris=refcc.eris)
mycc.t1, mycc.t2 = t1, t2
refcc.t1, refcc.t2 = t1r, t2r

myip = eom_uccsd.EOMIP(mycc)
myea = eom_uccsd.EOMEA(mycc)

refip = cc.eom_uccsd.EOMIP(refcc)
refea = cc.eom_uccsd.EOMEA(refcc)

imds = myip.make_imds()
imds.make_ea()

imdsr = refip.make_imds()
imdsr.make_ea()

class REFTEST(unittest.TestCase):
    def test_imds(self):
        for key in imds.__dict__.keys():
            if key[0] in ['F', 'W']:
                x = getattr(imds, key, None)
                y = getattr(imdsr,key, None)
                if x is not None and y is not None:
                    x = x.array.to_nparray()
                    self.assertTrue(numpy.linalg.norm(x - y)<1e-8)

    def test_matvec(self):
        vec = ctf.random.random([myip.vector_size()])
        vecnew = myip.matvec(vec, imds=imds)
        r1, r2 = myip.vector_to_amplitudes(vec)
        r1new, r2new = myip.vector_to_amplitudes(vecnew)
        r1r = [i.array.to_nparray() for i in r1]
        r2r = [i.array.to_nparray() for i in r2]
        vecr = refip.amplitudes_to_vector(r1r, r2r)
        vecnewr = refip.matvec(vecr, imds=imdsr)
        r1newr, r2newr = refip.vector_to_amplitudes(vecnewr)
        for i, j in zip(r1new+r2new, r1newr+r2newr):
            self.assertTrue(numpy.linalg.norm(i.array.to_nparray() - j)<1e-8)

        vec = ctf.random.random([myea.vector_size()])
        vecnew = myea.matvec(vec, imds=imds)
        r1, r2 = myea.vector_to_amplitudes(vec)
        r1new, r2new = myea.vector_to_amplitudes(vecnew)
        r1r = [i.array.to_nparray() for i in r1]
        r2r = [i.array.to_nparray() for i in r2]
        vecr = refea.amplitudes_to_vector(r1r, r2r)
        vecnewr = refea.matvec(vecr, imds=imdsr)
        r1newr, r2newr = refea.vector_to_amplitudes(vecnewr)
        for i, j in zip(r1new+r2new, r1newr+r2newr):
            self.assertTrue(numpy.linalg.norm(i.array.to_nparray() - j)<1e-8)

if __name__ == '__main__':
    print("eom_uccsd tests")
    unittest.main()
