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
import tempfile
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto, scf, ao2mo
from pyscf.tools import fcidump

mol = gto.Mole()
mol.atom = '''
N  0.0000000000   0.0000000000   0.0000000000
N  0.0000000000   0.0000000000   1.0977000000
           '''
mol.basis = 'sto-3g'
mol.symmetry = 1
mol.symmetry_subgroup = 'D2h'
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.verbose = 0
mol.build(0, 0)

mf = scf.RHF(mol)
mf.scf()

class KnowValues(unittest.TestCase):
    def test_from_chkfile(self):
        tmpfcidump = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fcidump.from_chkfile(tmpfcidump.name, mf.chkfile, tol=1e-15)

    def test_from_integral(self):
        tmpfcidump = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        h2 = ao2mo.full(mf._eri, mf.mo_coeff)
        fcidump.from_integrals(tmpfcidump.name, h1, h2, h1.shape[0],
                               mol.nelectron, tol=1e-15)

if __name__ == "__main__":
    print("Full Tests for fcidump")
    unittest.main()



