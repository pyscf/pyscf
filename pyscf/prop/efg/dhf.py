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

'''
Electric field gradients, nuclear quadrupolar coupling and Mossbauer
spectroscopy for relativistic 4-component DHF and DKS methods.
(In testing)
'''

import numpy
from pyscf import lib
from pyscf.prop.efg import rhf as rhf_efg

def kernel(method, efg_nuc=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** EFG for 4-component SCF methods (In testing) ********')
    mol = method.mol
    if efg_nuc is None:
        efg_nuc = range(mol.natm)

    c = lib.param.LIGHT_SPEED
    dm = method.make_rdm1()

    log.info('\nElectric Field Gradient Tensor Results')
    n2c = mol.nao_2c()
    coords = mol.atom_coords()
    aoLa, aoLb = mol.eval_gto('GTOval_spinor', coords)
    aoSa, aoSb = mol.eval_gto('GTOval_sp_spinor', coords)
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        # The electronic quadrupole operator (3 \vec{r} \vec{r} - r^2) / r^5
        with mol.with_rinv_origin(coords[atm_id]):
            ipipv = mol.intor('int1e_ipiprinv_spinor', 9).reshape(3,3,n2c,n2c)
            ipvip = mol.intor('int1e_iprinvip_spinor', 9).reshape(3,3,n2c,n2c)
            h1LL = ipipv + ipvip  # (nabla i | r/r^3 | j)
            h1LL = h1LL + h1LL.conj().transpose(0,1,3,2)
            trace = h1LL[0,0] + h1LL[1,1] + h1LL[2,2]
            h1LL[0,0] -= trace
            h1LL[1,1] -= trace
            h1LL[2,2] -= trace

            ipipv = mol.intor('int1e_ipipsprinvsp_spinor', 9).reshape(3,3,n2c,n2c)
            ipvip = mol.intor('int1e_ipsprinvspip_spinor', 9).reshape(3,3,n2c,n2c)
            h1SS = ipipv + ipvip  # (nabla i | r/r^3 | j)
            h1SS = h1SS + h1SS.conj().transpose(0,1,3,2)
            trace = h1SS[0,0] + h1SS[1,1] + h1SS[2,2]
            h1SS[0,0] -= trace
            h1SS[1,1] -= trace
            h1SS[2,2] -= trace

        fcLL = numpy.einsum('p,q->pq', aoLa[atm_id].conj(), aoLa[atm_id])
        fcLL+= numpy.einsum('p,q->pq', aoLb[atm_id].conj(), aoLb[atm_id])
        fcSS = numpy.einsum('p,q->pq', aoSa[atm_id].conj(), aoSa[atm_id])
        fcSS+= numpy.einsum('p,q->pq', aoSb[atm_id].conj(), aoSb[atm_id])

        fcsd = numpy.einsum('xyij,ji->xy', h1LL, dm[:n2c,:n2c])
        fcsd+= numpy.einsum('xyij,ji->xy', h1SS, dm[n2c:,n2c:]) * (.5/c)**2
        fc = numpy.einsum('ij,ji->', fcLL, dm[:n2c,:n2c])
        fc+= numpy.einsum('ij,ji->', fcSS, dm[n2c:,n2c:]) * (.5/c)**2
        efg_e = fcsd - 8*numpy.pi/3 * numpy.eye(3) * fc

        efg_nuc = rhf_efg._get_quad_nuc(mol, atm_id)
        v = efg_nuc - efg_e
        efg.append(v)

        rhf_efg._analyze(mol, atm_id, v.real, log)

    return numpy.asarray(efg).real

EFG = kernel

from pyscf import scf
scf.dhf.UHF.EFG = lib.class_as_method(EFG)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = 'ccpvdz'
    mol.unit = 'B'
    mol.build()

    mf = scf.DHF(mol).run()
    mf.EFG()
