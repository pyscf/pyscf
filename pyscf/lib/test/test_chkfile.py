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
import tempfile
from pyscf import lib, gto

class KnownValues(unittest.TestCase):
    def test_save_load_mol(self):
        mol = gto.M(atom=[['H', (0,0,i)] for i in range(8)],
                    basis='sto3g')
        fchk = tempfile.NamedTemporaryFile()
        lib.chkfile.save_mol(mol, fchk.name)
        mol1 = lib.chkfile.load_mol(fchk.name)
        self.assertTrue(numpy.all(mol1._atm == mol._atm))
        self.assertTrue(numpy.all(mol1._bas == mol._bas))
        self.assertTrue(numpy.all(mol1._env == mol._env))

    def test_save_load_arrays(self):
        fchk = tempfile.NamedTemporaryFile()
        a = numpy.eye(3)
        lib.chkfile.save(fchk.name, 'a', a)
        self.assertTrue(numpy.all(a == lib.chkfile.load(fchk.name, 'a')))

        a = [numpy.eye(3), numpy.eye(4)]
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue(isinstance(dat, list))
        self.assertTrue(numpy.all(a[1] == dat[1]))

        a = [[numpy.eye(4), numpy.eye(4)]]*2
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue(isinstance(dat, list))
        self.assertTrue(isinstance(dat[0], list))

        a = {'x':[numpy.eye(4), numpy.eye(4)],
             'y':[numpy.eye(4)]}
        lib.chkfile.save(fchk.name, 'a', a)
        dat = lib.chkfile.load(fchk.name, 'a')
        self.assertTrue('x' in dat)
        self.assertTrue('y' in dat)

if __name__ == "__main__":
    print("Full Tests for lib.chkfile")
    unittest.main()
