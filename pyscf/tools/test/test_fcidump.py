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
import tempfile

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = '''
    N  0.0000000000   0.0000000000   0.0000000000
    N  0.0000000000   0.0000000000   1.0977000000
               '''
    mol.basis = 'sto-3g'
    mol.symmetry = 'D2h'
    mol.charge = 0
    mol.spin = 0 #2*S; multiplicity-1
    mol.verbose = 0
    mol.build(0, 0)

    mf = mol.RHF(chkfile=tempfile.NamedTemporaryFile().name).run()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_from_chkfile(self):
        tmpfcidump = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fcidump.from_chkfile(tmpfcidump.name, mf.chkfile, tol=1e-15,
                             molpro_orbsym=True)

    def test_from_integral(self):
        tmpfcidump = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        h2 = ao2mo.full(mf._eri, mf.mo_coeff)
        fcidump.from_integrals(tmpfcidump.name, h1, h2, h1.shape[0],
                               mol.nelectron, tol=1e-15)

    def test_read(self):
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''&FCI NORB=4,
NELEC=4, MS2=0, ISYM=1,
ORBSYM=1,2,3,4,
&END
0.42 1 1 1 1
0.33 1 1 2 2
0.07 1 1 3 1
0.46 1 1 0 0
0.13 1 2 0 0
1.1  0 0 0 0
''')
            f.flush()
            result = fcidump.read(f.name)
        self.assertEqual(result['ISYM'], 1)

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write('''&FCI NORB=4, NELEC=4, MS2=0, ISYM=1,ORBSYM=1,2,3,4, &END
0.42 1 1 1 1
0.33 1 1 2 2
0.07 1 1 3 1
0.46 1 1 0 0
0.13 1 2 0 0
1.1  0 0 0 0
''')
            f.flush()
            result = fcidump.read(f.name)
        self.assertEqual(result['MS2'], 0)

    def test_to_scf(self):
        '''Test from_scf and to_scf'''
        tmpfcidump = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fcidump.from_scf(mf, tmpfcidump.name)
        mf1 = fcidump.to_scf(tmpfcidump.name)
        mf1.init_guess = mf.make_rdm1()
        mf1.kernel()
        self.assertTrue(abs(mf1.e_tot - mf.e_tot).max() < 1e-9)
        self.assertTrue(numpy.array_equal(mf.orbsym, mf1.orbsym))

    def test_to_scf_with_symmetry(self):
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfcidump:
            mol = gto.M(atom='H 0 0 0; H 1 0 0', symmetry=True)
            mf = mol.RHF().run()
            fcidump.from_scf(mf, tmpfcidump.name)
            mf = fcidump.to_scf(tmpfcidump.name)
            self.assertEqual(mf.mol.groupname, 'D2h')


if __name__ == "__main__":
    print("Full Tests for fcidump")
    unittest.main()
