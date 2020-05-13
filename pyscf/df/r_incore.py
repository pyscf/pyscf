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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto
from pyscf.df import incore

# (ij|L)
def aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=None, hermi=0):
    intor, comp = gto.moleintor._get_intor_and_comp(mol._add_suffix(intor), comp)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    ao_loc1 = mol.ao_loc_2c()
    ao_loc2 = auxmol.ao_loc_nr('ssc' in intor)
    nao = ao_loc1[-1]
    ao_loc = numpy.append(ao_loc1, ao_loc2[1:]+nao)
    out = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice,
                                  comp, aosym, ao_loc=ao_loc)
    return out

# (L|ij)
def aux_e1(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=1, hermi=0):
    raise NotImplementedError


def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e_spinor', aosym='s1', int2c='int2c2e_sph', comp=1,
                 verbose=0):
    return incore.cholesky_eri_debug(mol, auxbasis, auxmol, int3c, aosym, int2c,
                                     comp, verbose, aux_e2)


if __name__ == '__main__':
    from pyscf import lib
    from pyscf import scf
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    cderi = (cholesky_eri(mol, int3c='int3c2e_spinor', verbose=5),
             cholesky_eri(mol, int3c='int3c2e_spsp1_spinor', verbose=5))
    n2c = mol.nao_2c()
    c2 = .5 / lib.param.LIGHT_SPEED
    def fjk(mol, dm, *args, **kwargs):
        # dm is 4C density matrix
        cderi_ll = cderi[0].reshape(-1,n2c,n2c)
        cderi_ss = cderi[1].reshape(-1,n2c,n2c)
        vj = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)
        vk = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)

        rho =(numpy.dot(cderi[0], dm[:n2c,:n2c].T.reshape(-1))
            + numpy.dot(cderi[1], dm[n2c:,n2c:].T.reshape(-1)*c2**2))
        vj[:n2c,:n2c] = numpy.dot(rho, cderi[0]).reshape(n2c,n2c)
        vj[n2c:,n2c:] = numpy.dot(rho, cderi[1]).reshape(n2c,n2c) * c2**2

        v1 = numpy.einsum('pij,jk->pik', cderi_ll, dm[:n2c,:n2c])
        vk[:n2c,:n2c] = numpy.einsum('pik,pkj->ij', v1, cderi_ll)
        v1 = numpy.einsum('pij,jk->pik', cderi_ss, dm[n2c:,n2c:])
        vk[n2c:,n2c:] = numpy.einsum('pik,pkj->ij', v1, cderi_ss) * c2**4
        v1 = numpy.einsum('pij,jk->pik', cderi_ll, dm[:n2c,n2c:])
        vk[:n2c,n2c:] = numpy.einsum('pik,pkj->ij', v1, cderi_ss) * c2**2
        vk[n2c:,:n2c] = vk[:n2c,n2c:].T.conj()
        return vj, vk

    mf = scf.DHF(mol)
    mf.get_jk = fjk
    mf.direct_scf = False
    ehf1 = mf.scf()
    print(ehf1, -76.08073868516945)

    cderi = cderi[0].reshape(-1,n2c,n2c)
    print(numpy.allclose(cderi, cderi.transpose(0,2,1).conj()))
