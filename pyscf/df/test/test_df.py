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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''O     0    0.       0.
              1     0    -0.757   0.587
              1     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_ao2mo(self):
        dfobj = df.DF(mol)
        # force DF intermediates to be saved on disk
        dfobj.max_memory = 0.01

        # Initialize _call_count, to test DF.prange function
        dfobj._call_count = 0

        # dfobj.build is called in dfobj.get_naoaux()
        self.assertEqual(dfobj.get_naoaux(), 116)

        #dfobj.build()
        cderi = dfobj._cderi

        nao = mol.nao_nr()
        eri0 = dfobj.get_eri()
        numpy.random.seed(1)
        mos = numpy.random.random((nao,nao*10))
        mos = (mos[:,:5], mos[:,5:11], mos[:,3:9], mos[:,2:4])
        mo_eri0 = ao2mo.kernel(eri0, mos)

        mo_eri1 = dfobj.ao2mo(mos)
        self.assertAlmostEqual(abs(mo_eri0-mo_eri1).max(), 0, 9)

        mo = numpy.random.random((nao,nao))
        mo_eri0 = ao2mo.kernel(eri0, mo)
        mo_eri1 = dfobj.ao2mo(mo)
        self.assertAlmostEqual(abs(mo_eri0-mo_eri1).max(), 0, 9)

    def test_cderi_to_save(self):
        with open(os.devnull, 'w') as f:
            ftmp = tempfile.NamedTemporaryFile()
            dfobj = df.DF(mol)
            dfobj.auxmol = df.addons.make_auxmol(mol, 'weigend')
            dfobj.verbose = 5
            dfobj.stdout = f
            dfobj._cderi_to_save = ftmp.name
            dfobj._cderi = 'abc'
            dfobj.kernel()
            eri0 = dfobj.get_eri()
        dfobj = df.DF(mol)
        dfobj._cderi = ftmp.name
        eri1 = dfobj.get_eri()
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 9)

    def test_init_denisty_fit(self):
        from pyscf.df import df_jk
        from pyscf import cc
        from pyscf.cc import dfccsd
        self.assertTrue(isinstance(df.density_fit(scf.RHF(mol)), df_jk._DFHF))
        self.assertTrue(isinstance(df.density_fit(cc.CCSD(scf.RHF(mol))),
                                   dfccsd.RCCSD))


if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

