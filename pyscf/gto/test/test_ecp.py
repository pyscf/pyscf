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


def setUpModule():
    global mol, mol1, mol2, cu1_basis
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
        self.assertAlmostEqual(lib.fp(mol.intor('ECPscalar')), -0.19922134780248762, 9)
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
                with mol.with_rinv_at_nucleus(0):
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

    def test_pp_int(self):
        from pyscf import gto, scf
        from pyscf.pbc import gto as pbcgto
        from pyscf.pbc import scf as pbcscf
        from pyscf.pbc import df
        cell = pbcgto.Cell()
        cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                      'C' :'gth-szv',}
        cell.pseudo = {'C':'gth-pade',
                       'He': pbcgto.pseudo.parse('''He
        2
         0.40000000    3    -1.98934751    -0.75604821    0.95604821
        2
         0.29482550    3     1.23870466    .855         .3
                                           .71         -1.1
                                                        .9
         0.32235865    2     2.25670239    -0.39677748
                                            0.93894690
                                                     ''')}
        cell.a = numpy.eye(3)
        cell.dimension = 0
        cell.build()
        mol = cell.to_mol()

        hcore = scf.RHF(mol).get_hcore()
        mydf = df.AFTDF(cell)
        ref = mydf.get_pp() + mol.intor('int1e_kin')
        #FIXME: error seems big
        self.assertAlmostEqual(abs(hcore-ref).max(), 0, 2)

        mf = pbcscf.RHF(cell)
        mf.with_df = mydf
        mf.run()
        e_ref = mf.e_tot

        e_tot = scf.RHF(mol).run().e_tot
        self.assertAlmostEqual(abs(e_ref-e_tot).max(), 0, 5)

    def test_scalar_vs_int1e_rinv(self):
        mol = gto.M(atom='''
                    Na 0.5 0.5 0.
                    H  1.0 0.  0.2
                    ''',
                    basis={'Na': [(0, (1, 1)), (1, (4, 1)), (2, (1, 1))],
                           'H': 'ccpvtz'},
                    ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 8
Na ul
1      0.    -3.
''')})
        mat = mol.intor('ECPscalar')
        with mol.with_rinv_orig(mol.atom_coord(0)):
            ref = mol.intor('int1e_rinv')*-3
        self.assertAlmostEqual(abs(mat-ref).max(), 0, 9)

    def test_so_vs_int1e_rinv(self):
        mol = gto.M(atom='''
                    Na 0.5 0.5 0.
                    ''',
                    charge=1,
                    basis={'Na': [(0, (1, 1)), (1, (4, 1)), (1, (1, 1)), (2, (1, 1))]},
                    ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 8
Na S
0      0.     0     0
1      0.    -3.    -3.
Na P
1      0.    -3.    -3.
Na D
1      0.    -3.    -3.
Na F
1      0.    -3.    -3.
''')})
        u = mol.sph2spinor_coeff()
        ref = numpy.einsum('sxy,spq,xpi,yqj->ij', lib.PauliMatrices,
                           mol.intor('int1e_inuc_rxp'), u.conj(), u)

        mat = mol.intor('ECPso_spinor')
        self.assertAlmostEqual(abs(ref-mat).max(), 0, 11)

        mat = numpy.einsum('sxy,spq,xpi,yqj->ij', lib.PauliMatrices,
                           mol.intor('ECPso'), u.conj(), u)
        self.assertAlmostEqual(abs(ref-mat).max(), 0, 11)

    def test_ecp_with_so_data(self):
        mol = gto.M(atom='Ce 0 0 0', basis=[[0, [1, 1]]],
                    ecp='''
        Ce nelec 28
        Ce ul
        1     148.23398733 8.58586082
        Ce P
        0     0.25454001 5.00000000
        ''')
        mol2 = gto.M(atom='Ce 0 0 0', basis=[[0, [1, 1]]],
                     ecp='''
        Ce nelec 28
        Ce ul
        1     148.23398733 8.58586082 -0.69175758
        Ce P
        0     0.25454001 5.00000000 0.00000000
        ''')
        v1 = mol.intor('ECPscalar')
        v2 = mol2.intor('ECPscalar')
        self.assertAlmostEqual(v1[0,0], 0.1823961651083, 12)
        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 12)

    def test_ecp_grad1(self):
        mol = gto.M(atom='Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.050',
                    basis='lanl2dz', ecp = 'lanl2dz', unit='B')
        with mol.with_rinv_at_nucleus(0):
            iprinv0 = mol.intor('ECPscalar_iprinv')
        with mol.with_rinv_at_nucleus(1):
            iprinv1 = mol.intor('ECPscalar_iprinv')
        ipnuc = mol.intor('ECPscalar_ipnuc')
        self.assertAlmostEqual(abs(iprinv0 + iprinv1 - ipnuc).max(), 0, 12)

        aoslices = mol.aoslice_by_atom()
        atm_id = 1
        shl0, shl1, p0, p1 = aoslices[atm_id]
        mat0 = iprinv1
        mat0[:,p0:p1] -= ipnuc[:,p0:p1]
        mat0 = mat0 + mat0.transpose(0,2,1)

        mat1 = mol.set_geom_('Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.049').intor('ECPscalar')
        mat2 = mol.set_geom_('Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.051').intor('ECPscalar')
        self.assertAlmostEqual(abs(mat0[2] - (mat2 - mat1) / 0.002).max(), 0, 5)

    def test_ecp_hessian1(self):
        mol = gto.M(atom='Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.050',
                    basis='lanl2dz', ecp = 'lanl2dz', unit='B')
        with mol.with_rinv_at_nucleus(1):
            rinv1 = mol.intor('ECPscalar_ipiprinv', comp=9)
            rinv1+= mol.intor('ECPscalar_iprinvip', comp=9)
        ipipnuc = mol.intor('ECPscalar_ipipnuc', comp=9)
        ipnucip = mol.intor('ECPscalar_ipnucip', comp=9)

        aoslices = mol.aoslice_by_atom()
        atm_id = 1
        shl0, shl1, p0, p1 = aoslices[atm_id]
        mat0 = rinv1
        mat0[:,p0:p1] -= ipipnuc[:,p0:p1]
        mat0[:,:,p0:p1] -= ipnucip[:,:,p0:p1]

        nao = mol.nao
        mat1 = mol.set_geom_('Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.049').intor('ECPscalar_ipnuc')
        mat2 = mol.set_geom_('Na, 0.00, 0.00, 0.00; Cl, 0.00, 0.00, 2.051').intor('ECPscalar_ipnuc')
        self.assertAlmostEqual(abs(mat0.reshape(3,3,nao,nao)[:,2] - (mat2 - mat1) / 0.002).max(), 0, 5)

    def test_ecp_f_in_core(self):
        mol = gto.M(atom='Eu1, 0.00, 0.00, 0.00',
                    basis={'Eu': gto.basis.parse('''
Eu    S
    0.749719700E+01   -0.288775043E+00
    0.617255600E+01    0.708008105E+00
    0.260816600E+01   -0.136569920E+01
Eu    S
    0.530389000E+00    0.100000000E+01
Eu    S
    0.254033000E+00    0.100000000E+01
Eu    S
    0.522020000E-01    0.100000000E+01
Eu    S
    0.221100000E-01    0.100000000E+01
Eu    P
    0.399434200E+01    0.110821693E+01
    0.350361700E+01   -0.152518191E+01
    0.722399000E+00    0.119866293E+01
Eu    P
    0.324354000E+00    0.100000000E+01
Eu    P
    0.127842000E+00    0.100000000E+01
Eu    P
    0.330280000E-01    0.100000000E+01
Eu    D
    0.206170800E+01   -0.127297005E+00
    0.967971000E+00    0.377785014E+00
    0.369101000E+00    0.765795028E+00
Eu    D
    0.128958000E+00    0.100000000E+01
Eu    D
    0.419270000E-01    0.100000000E+01
                    ''')},
                    ecp={'Eu': gto.basis.parse_ecp('''
Eu nelec  53
Eu ul
2      1.0000000000        0.0000000000
Eu S
2      5.1852000000      172.7978960000
2      2.5926000000      -10.0922600000
Eu P
2      4.3588000000      111.3150270000
2      2.1794000000       -3.4025580000
Eu D
2      2.8902000000       41.8677290000
2      1.4451000000       -1.2874330000
Eu F
2      5.3988000000      -63.6010500000
                    ''')}, charge=2, verbose=0)
        mf = scf.RHF(mol)
        mf.get_init_guess()
        self.assertEqual(mol.ao_labels()[0], '0 Eu1 5s    ')
        self.assertAlmostEqual(lib.fp(mf.get_hcore()), 22.59028455662168)

    def test_ecp_f_in_valence(self):
        mol = gto.M(atom='U, 0.00, 0.00, 0.00',
                    basis={'U': 'crenbl'}, ecp={'U': 'crenbl'},
                    charge=3, spin=3, verbose=0)
        mf = scf.ROHF(mol)
        mf.get_init_guess()
        self.assertEqual(mol.ao_labels()[40], '0 U 5f-3  ')
        self.assertAlmostEqual(lib.fp(mf.get_hcore()), -55.38627201912257)


if __name__ == '__main__':
    print("Full Tests for ECP")
    unittest.main()
