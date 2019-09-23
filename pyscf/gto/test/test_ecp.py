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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib


cu1_basis = gto.basis.parse('''
 H    S
       1.8000000              1.0000000
 H    S
       2.8000000              0.0210870             -0.0045400              0.0000000
       1.3190000              0.3461290             -0.1703520              0.0000000
       0.9059000              0.0393780              0.1403820              1.0000000
 H    P
       2.1330000              0.0868660              0.0000000
       1.2000000              0.0000000              0.5000000
       0.3827000              0.5010080              1.0000000
 H    D
       0.3827000              1.0000000
 H    F
       2.1330000              0.1868660              0.0000000
       0.3827000              0.2010080              1.0000000
                               ''')

mol = gto.M(atom='''
Cu1 0. 0. 0.
Cu 0. 1. 0.
He 1. 0. 0.
''',
            basis={'Cu':'lanl2dz', 'Cu1': cu1_basis, 'He':'sto3g'},
            ecp = {'cu':'lanl2dz'})

mol1 = gto.M(atom='''
Cu1 0.  0.  0.
Cu 0. 1. 0.
He 1. 0. 0.
Ghost-Cu1 0.  0.  0.0001
''',
             basis={'Cu':'lanl2dz', 'Cu1': cu1_basis, 'He':'sto3g'},
             ecp = {'cu':'lanl2dz'})

mol2 = gto.M(atom='''
Cu1 0.  0.  0.
Cu 0. 1. 0.
He 1. 0. 0.
Ghost-Cu1 0.  0. -0.0001
''',
             basis={'Cu':'lanl2dz', 'Cu1': cu1_basis, 'He':'sto3g'},
             ecp = {'cu':'lanl2dz'})

def tearDownModule():
    global mol, mol1, mol2, cu1_basis
    del mol, mol1, mol2, cu1_basis

class KnownValues(unittest.TestCase):
    def test_ecp_by_shell(self):
        for i in (0,2,3,6,9):
            for j in (1,2,3,5,6):
                ref = mol.intor_by_shell('ECPscalar_sph', (i,j))
                dat = gto.ecp.type1_by_shell(mol, (i, j))
                dat+= gto.ecp.type2_by_shell(mol, (i, j))
                self.assertAlmostEqual(abs(ref-dat).max(), 0, 12)

                ref = mol.intor_by_shell('ECPscalar_cart', (i,j))
                dat = gto.ecp.type1_by_shell(mol, (i, j), cart=True)
                dat+= gto.ecp.type2_by_shell(mol, (i, j), cart=True)
                self.assertAlmostEqual(abs(ref-dat).max(), 0, 12)

    def test_nr_rhf(self):
        mol = gto.M(atom='Na 0. 0. 0.;  H  0.  0.  1.',
                    basis={'Na':'lanl2dz', 'H':'sto3g'},
                    ecp = {'Na':'lanl2dz'},
                    verbose=0)
        self.assertAlmostEqual(lib.finger(mol.intor('ECPscalar')), -0.19922134780248762, 9)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.45002315563472206, 10)

    def test_bfd(self):
        mol = gto.M(atom='H 0. 0. 0.',
                    basis={'H':'bfd-vdz'},
                    ecp = {'H':'bfd-pp'},
                    spin = 1,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.499045, 6)

        mol = gto.M(atom='Na 0. 0. 0.',
                    basis={'Na':'bfd-vtz'},
                    ecp = {'Na':'bfd-pp'},
                    spin = 1,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.181799, 6)

        mol = gto.M(atom='Mg 0. 0. 0.',
                    basis={'Mg':'bfd-vtz'},
                    ecp = {'Mg':'bfd-pp'},
                    spin = 0,
                    verbose=0)
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -0.784579, 6)

#        mol = gto.M(atom='Ne 0. 0. 0.',
#                    basis={'Ne':'bfd-vdz'},
#                    ecp = {'Ne':'bfd-pp'},
#                    verbose=0)
#        mf = scf.RHF(mol)
#        self.assertAlmostEqual(mf.kernel(), -34.709059, 6)

    def test_ecp_grad(self):
        aoslices = mol.aoslice_nr_by_atom()
        ish0, ish1 = aoslices[0][:2]
        for i in range(ish0, ish1):
            for j in range(mol.nbas):
                shls = (i,j)
                shls1 = (shls[0] + mol.nbas, shls[1])
                ref = (mol1.intor_by_shell('ECPscalar_cart', shls1) -
                       mol2.intor_by_shell('ECPscalar_cart', shls1)) / 0.0002 * lib.param.BOHR
                dat = mol.intor_by_shell('ECPscalar_ipnuc_cart', shls, comp=3)
                self.assertAlmostEqual(abs(-dat[2]-ref).max(), 0, 4)

    def test_ecp_iprinv(self):
        mol = gto.M(atom='''
        Cu 0. 0. 0.
        H  1. 0. 0.
        ''',
                    basis={'Cu':'lanl2dz', 'H':'ccpvdz'},
                    ecp = {'cu':'lanl2dz'})
        mol1 = gto.M(atom='''
        Cu 0. 0. 0.
        H  1. 0. 0.
        Ghost-Cu 0.  0.  0.0001
        ''',
                    basis={'Cu':'lanl2dz', 'H':'ccpvdz'},
                    ecp = {'cu':'lanl2dz'})
        mol2 = gto.M(atom='''
        Cu 0. 0. 0.
        H  1. 0. 0.
        Ghost-Cu 0.  0. -0.0001
        ''',
                    basis={'Cu':'lanl2dz', 'H':'ccpvdz'},
                    ecp = {'cu':'lanl2dz'})
        aoslices = mol.aoslice_nr_by_atom()
        ish0, ish1 = aoslices[0][:2]
        for i in range(ish0, ish1):
            for j in range(mol.nbas):
                shls = (i,j)
                shls1 = (shls[0] + mol.nbas, shls[1])
                ref = (mol1.intor_by_shell('ECPscalar_cart', shls1) -
                       mol2.intor_by_shell('ECPscalar_cart', shls1)) / 0.0002 * lib.param.BOHR
                with mol.with_rinv_as_nucleus(0):
                    dat = mol.intor_by_shell('ECPscalar_iprinv_cart', shls, comp=3)
                self.assertAlmostEqual(abs(-dat[2]-ref).max(), 0, 4)

    def test_ecp_hessian(self):
        aoslices = mol.aoslice_nr_by_atom()
        ish0, ish1 = aoslices[0][:2]
        for i in range(ish0, ish1):
            for j in range(mol.nbas):
                shls = (i,j)
                shls1 = (shls[0] + mol.nbas, shls[1])
                ref =-(mol1.intor_by_shell('ECPscalar_ipnuc_cart', shls1, comp=3) -
                       mol2.intor_by_shell('ECPscalar_ipnuc_cart', shls1, comp=3)) / 0.0002 * lib.param.BOHR
                dat = mol.intor_by_shell('ECPscalar_ipipnuc_cart', shls, comp=9)
                di, dj = dat.shape[1:]
                dat = dat.reshape(3,3,di,dj)
                self.assertAlmostEqual(abs(dat[2]-ref).max(), 0, 3)

        for i in range(mol.nbas):
            for j in range(ish0, ish1):
                shls = (i,j)
                shls1 = (shls[0], shls[1] + mol.nbas)
                ref =-(mol1.intor_by_shell('ECPscalar_ipnuc_cart', shls1, comp=3) -
                       mol2.intor_by_shell('ECPscalar_ipnuc_cart', shls1, comp=3)) / 0.0002 * lib.param.BOHR
                dat = mol.intor_by_shell('ECPscalar_ipnucip_cart', shls, comp=9)
                di, dj = dat.shape[1:]
                dat = dat.reshape(3,3,di,dj)
                self.assertAlmostEqual(abs(dat[:,2]-ref).max(), 0, 3)


if __name__ == '__main__':
    print("Full Tests for ECP")
    unittest.main()

