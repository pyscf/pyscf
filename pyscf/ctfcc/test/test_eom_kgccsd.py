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
from pyscf.pbc import gto, scf, cc
from pyscf.ctfcc import kccsd, eom_kccsd_ghf

cell = gto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.mesh = [13,13,13]
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000
'''
cell.spin = 2
cell.output = '/dev/null'
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KUHF(cell,kpts, exxdiv=None)
mf.max_cycle = 1
mf.kernel()
mf = mf.to_ghf(mf)
mycc = kccsd.KGCCSD(mf)
mycc.max_cycle=1
mycc.kernel()

refcc = cc.KGCCSD(mf)
refcc.max_cycle=1
refcc.kernel()

myip = eom_kccsd_ghf.EOMIP(mycc)
myea = eom_kccsd_ghf.EOMEA(mycc)

refip = cc.eom_kccsd_ghf.EOMIP(refcc)
refea = cc.eom_kccsd_ghf.EOMEA(refcc)

imds = myip.make_imds()
imds.make_ea()

imdsr = refip.make_imds()
imdsr.make_ea()

class REFTEST(unittest.TestCase):
    def test_imds(self):
        for i in imds.__dict__.keys():
            if i[0].lower() != 'w': continue
            wx = getattr(imds, i)
            wy = getattr(imdsr, i)
            if wx is not None and wy is not None:
                self.assertTrue(numpy.linalg.norm(wx.array.to_nparray()-wy)<1e-5)

    def test_amp_vec_ip(self):
        kshift=2
        vecsize=  myip.vector_size()
        vec = ctf.random.random([vecsize]) + 0.2j * ctf.random.random([vecsize])
        r1, r2 = myip.vector_to_amplitudes(vec, kshift)
        r11, r21 = myip.vector_to_amplitudes(myip.amplitudes_to_vector(r1, r2), kshift)

        r1r = r1.array.to_nparray()
        r2r = r2.array.to_nparray()
        r11r, r21r = refip.vector_to_amplitudes(refip.amplitudes_to_vector(r1r, r2r, kshift), kshift)

        self.assertTrue(numpy.linalg.norm(r11.array.to_nparray()-r11r)<1e-10)
        self.assertTrue(numpy.linalg.norm(r21.array.to_nparray()-r21r)<1e-10)

    def test_amp_vec_ea(self):
        kshift=1
        vecsize=  myea.vector_size()
        vec = ctf.random.random([vecsize]) + 0.2j * ctf.random.random([vecsize])
        r1, r2 = myea.vector_to_amplitudes(vec, kshift)
        r11, r21 = myea.vector_to_amplitudes(myea.amplitudes_to_vector(r1, r2), kshift)

        r1r = r1.array.to_nparray()
        r2r = r2.array.to_nparray()
        r11r, r21r = refea.vector_to_amplitudes(refea.amplitudes_to_vector(r1r, r2r, kshift), kshift)

        self.assertTrue(numpy.linalg.norm(r11.array.to_nparray()-r11r)<1e-10)
        self.assertTrue(numpy.linalg.norm(r21.array.to_nparray()-r21r)<1e-10)

    def test_matvec_ip(self):
        kshift=2
        vecsize=  myip.vector_size()
        vec = ctf.random.random([vecsize]) + 0.2j * ctf.random.random([vecsize])
        r1, r2 = myip.vector_to_amplitudes(vec, kshift)
        r11, r21 = myip.vector_to_amplitudes(myip.matvec(vec, kshift, imds=imds), kshift)

        r1r = r1.array.to_nparray()
        r2r = r2.array.to_nparray()
        vecr = refip.amplitudes_to_vector(r1r, r2r, kshift)
        r11r, r21r = refip.vector_to_amplitudes(refip.matvec(vecr, kshift, imds=imdsr), kshift)
        self.assertTrue(numpy.linalg.norm(r11.array.to_nparray()-r11r)<1e-10)
        self.assertTrue(numpy.linalg.norm(r21.array.to_nparray()-r21r)<1e-10)

    def test_matvec_ea(self):
        kshift=1
        vecsize=  myea.vector_size()
        vec = ctf.random.random([vecsize]) + 0.2j * ctf.random.random([vecsize])
        r1, r2 = myea.vector_to_amplitudes(vec, kshift)
        r11, r21 = myea.vector_to_amplitudes(myea.matvec(vec, kshift, imds=imds), kshift)
        r1r = r1.array.to_nparray()
        r2r = r2.array.to_nparray()
        vecr = refea.amplitudes_to_vector(r1r, r2r, kshift)
        r11r, r21r = refea.vector_to_amplitudes(refea.matvec(vecr, kshift, imds=imdsr), kshift)

        self.assertTrue(numpy.linalg.norm(r11.array.to_nparray()-r11r)<1e-10)
        self.assertTrue(numpy.linalg.norm(r21.array.to_nparray()-r21r)<1e-10)

if __name__ == '__main__':
    print("eom_kccsd_ghf tests")
    unittest.main()
