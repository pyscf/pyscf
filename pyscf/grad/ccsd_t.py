#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

from pyscf import lib
from pyscf.cc import ccsd_t_rdm as ccsd_t_rdm
from pyscf.grad import ccsd as ccsd_grad

# Only works with canonical orbitals
def grad_elec(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              verbose=lib.logger.INFO):
    mycc = cc_grad.base
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = mycc.ao2mo()
    d1 = ccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris,
                                          for_grad=True)
    fd2intermediate = lib.H5TmpFile()
    d2 = ccsd_t_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, eris,
                                    fd2intermediate, True)
    cc_grad = ccsd_grad.Gradients(mycc)
    de = ccsd_grad.grad_elec(cc_grad, t1, t2, l1, l2, eris, atmlst,
                             d1, d2, verbose)
    return de

class Gradients(ccsd_grad.Gradients):
    grad_elec = grad_elec


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc
    from pyscf.cc import ccsd_t
    from pyscf.cc import ccsd_t_lambda as ccsd_t_lambda

    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.    )],
            [1   , (0. ,-0.757  ,-0.587)],
            [1   , (0. , 0.757  ,-0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf = mf.scf()

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)
    print(g1)

    myccs = mycc.as_scanner()
    mol.atom[0] = ["O" , (0., 0., 0.001)]
    mol.build(0, 0)
    e1 = myccs(mol)
    e1 += myccs.ccsd_t()
    mol.atom[0] = ["O" , (0., 0.,-0.001)]
    mol.build(0, 0)
    e2 = myccs(mol)
    e2 += myccs.ccsd_t()
    print(g1[0,2], (e1-e2)/0.002*lib.param.BOHR)
#O      0.0000000000            0.0000000000           -0.0112045345
#H      0.0000000000            0.0234464201            0.0056022672
#H      0.0000000000           -0.0234464201            0.0056022672


    mol = gto.M(
        verbose = 0,
        atom = '''
H         -1.90779510     0.92319522     0.08700656
H         -1.08388168    -1.61405643    -0.07315086
H          2.02822318    -0.61402169     0.09396693
H          0.96345360     1.30488291    -0.10782263
               ''',
        unit='bohr',
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf0+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)
    print(g1)

    myccs = mycc.as_scanner()
    mol.atom = '''
H         -1.90679510     0.92319522     0.08700656
H         -1.08388168    -1.61405643    -0.07315086
H          2.02822318    -0.61402169     0.09396693
H          0.96345360     1.30488291    -0.10782263
               '''
    mol.build(0, 0)
    e1 = myccs(mol)
    e1 += myccs.ccsd_t()
    mol.atom = '''
H         -1.90879510     0.92319522     0.08700656
H         -1.08388168    -1.61405643    -0.07315086
H          2.02822318    -0.61402169     0.09396693
H          0.96345360     1.30488291    -0.10782263
               '''
    mol.build(0, 0)
    e2 = myccs(mol)
    e2 += myccs.ccsd_t()
    print(g1[0,0], (e1-e2)/0.002)
#CCSD
#H   0.0113620114            0.0664344363            0.0029855587
#H   0.0528858926           -0.0483942979           -0.0033960631
#H   0.0109676543           -0.0827248466            0.0074005299
#H  -0.0752155583            0.0646847082           -0.0069900255
#
#CCSD(T) gradient:
#
#H   0.0112264011            0.0658917731            0.0029936671
#H   0.0525667206           -0.0481602008           -0.0033751503
#H   0.0107197424           -0.0823677005            0.0073804163
#H  -0.0745128642            0.0646361282           -0.0069989330
