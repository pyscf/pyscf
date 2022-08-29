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

'''
General JK contraction function for
* arbitrary integrals
* 4 different molecules
* multiple density matrices
* arbitrary basis subset for the 4 indices
'''

import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf


def get_jk(mols, dms, scripts=['ijkl,ji->kl'], intor='int2e_sph',
           aosym='s1', comp=None, hermi=0, shls_slice=None,
           verbose=logger.WARN, vhfopt=None):
    '''Compute J/K matrices for the given density matrix

    Args:
        mols : an instance of :class:`Mole` or a list of `Mole` objects

        dms : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether the returned J (K) matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

        intor : str
            2-electron integral name.  See :func:`getints` for the complete
            list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl)

        comp : int
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
        scripts : string or a list of strings
            Contraction description (following numpy.einsum convention) based on
            letters [ijkl].  Each script will be one-to-one applied to each
            entry of dms.  So it must have the same number of elements as the
            dms, len(scripts) == len(dms).
        shls_slice : 8-element list
            (ish_start, ish_end, jsh_start, jsh_end, ksh_start, ksh_end, lsh_start, lsh_end)

    Returns:
        Depending on the number of density matrices, the function returns one
        J/K matrix or a list of J/K matrices (the same number of entries as the
        input dms).
        Each JK matrices may be a 2D array or 3D array if the AO integral
        has multiple components.

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 -.5 0; H 0 .5 0', basis='cc-pvdz')
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((nao,nao))
    >>> # Default, Coulomb matrix
    >>> vj = get_jk(mol, dm)
    >>> # Coulomb matrix with 8-fold permutation symmetry for AO integrals
    >>> vj = get_jk(mol, dm, 'ijkl,ji->kl', aosym='s8')
    >>> # Exchange matrix with 8-fold permutation symmetry for AO integrals
    >>> vk = get_jk(mol, dm, 'ijkl,jk->il', aosym='s8')
    >>> # Compute coulomb and exchange matrices together
    >>> vj, vk = get_jk(mol, (dm,dm), ('ijkl,ji->kl','ijkl,li->kj'), aosym='s8')
    >>> # Analytical gradients for coulomb matrix
    >>> j1 = get_jk(mol, dm, 'ijkl,lk->ij', intor='int2e_ip1_sph', aosym='s2kl', comp=3)

    >>> # contraction across two molecules
    >>> mol1 = gto.M(atom='He 2 0 0', basis='6-31g')
    >>> nao1 = mol1.nao_nr()
    >>> dm1 = numpy.random.random((nao1,nao1))
    >>> # Coulomb interaction between two molecules, note 4-fold symmetry can be applied
    >>> jcross = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', aosym='s4')
    >>> ecoul = numpy.einsum('ij,ij', jcross, dm1)
    >>> # Exchange interaction between two molecules, no symmetry can be used
    >>> kcross = get_jk((mol1,mol,mol,mol1), dm, scripts='ijkl,jk->il')
    >>> ex = numpy.einsum('ij,ji', kcross, dm1)

    >>> # Analytical gradients for coulomb matrix between two molecules
    >>> jcros1 = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
    >>> # Analytical gradients for coulomb interaction between 1s density and the other molecule
    >>> jpart1 = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3,
    ...                 shls_slice=(0,1,0,1,0,mol.nbas,0,mol.nbas))
    '''
    if isinstance(mols, (tuple, list)):
        intor, comp = gto.moleintor._get_intor_and_comp(mols[0]._add_suffix(intor), comp)
        assert (len(mols) == 4)
        assert (mols[0].cart == mols[1].cart == mols[2].cart == mols[3].cart)
        if shls_slice is None:
            shls_slice = numpy.array([(0, mol.nbas) for mol in mols])
        else:
            shls_slice = numpy.asarray(shls_slice).reshape(4,2)

        # concatenate unique mols and build corresponding shls_slice
        mol_ids = [id(mol) for mol in mols]
        atm, bas, env = mols[0]._atm, mols[0]._bas, mols[0]._env
        bas_start = numpy.zeros(4, dtype=int)
        for m in range(1,4):
            first = mol_ids.index(mol_ids[m])
            if first == m:  # the unique mol, not repeated in mols
                bas_start[m] = bas.shape[0]
                atm, bas, env = gto.conc_env(atm, bas, env, mols[m]._atm,
                                             mols[m]._bas, mols[m]._env)
            else:
                bas_start[m] = bas_start[first]
            shls_slice[m] += bas_start[m]
        shls_slice = shls_slice.flatten()
    else:
        intor, comp = gto.moleintor._get_intor_and_comp(mols._add_suffix(intor), comp)
        atm, bas, env = mols._atm, mols._bas, mols._env
        if shls_slice is None:
            shls_slice = (0, mols.nbas) * 4

    single_script = isinstance(scripts, str)
    if single_script:
        scripts = [scripts]
    # Check if letters other than ijkl were provided.
    if set(''.join(scripts[:4])).difference('ijkl,->as12'):
        # Translate these letters to ijkl if possible
        scripts = [script.translate({ord(script[0]): 'i',
                                     ord(script[1]): 'j',
                                     ord(script[2]): 'k',
                                     ord(script[3]): 'l'})
                   for script in scripts]
        if set(''.join(scripts[:4])).difference('ijkl,->as12'):
            raise RuntimeError('Scripts unsupported %s' % scripts)

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = [dms]
    assert (len(scripts) == len(dms))

    #format scripts
    descript = []
    for script in scripts:
        dmsym, vsym = script.lower().split(',')[1].split('->')
        if vsym[:2] in ('a2', 's2', 's1'):
            descript.append(dmsym + '->' + vsym)
        elif hermi == 0:
            descript.append(dmsym + '->s1' + vsym)
        else:
            descript.append(dmsym + '->s2' + vsym)

    vs = _vhf.direct_bindm(intor, aosym, descript, dms, comp, atm, bas, env,
                           vhfopt=vhfopt, shls_slice=shls_slice)
    if hermi != 0:
        for v in vs:
            if v.ndim == 3:
                for vi in v:
                    lib.hermi_triu(vi, hermi, inplace=True)
            else:
                lib.hermi_triu(v, hermi, inplace=True)

    if single_script:
        vs = vs[0]
    return vs

jk_build = get_jk


if __name__ == '__main__':
    mol = gto.M(atom='H 0 -.5 0; H 0 .5 0', basis='cc-pvdz')

    nao = mol.nao_nr()
    dm = numpy.random.random((nao,nao))
    eri0 = mol.intor('int2e_sph').reshape((nao,)*4)
    vj = get_jk(mol, dm, 'ijkl,ji->kl')
    print(numpy.allclose(vj, numpy.einsum('ijkl,ji->kl', eri0, dm)))
    vj = get_jk(mol, dm, 'ijkl,ji->kl', aosym='s8')
    print(numpy.allclose(vj, numpy.einsum('ijkl,ji->kl', eri0, dm)))
    vk = get_jk(mol, dm, 'ijkl,jk->il', aosym='s8')
    print(numpy.allclose(vk, numpy.einsum('ijkl,jk->il', eri0, dm)))
    vj, vk = get_jk(mol, (dm,dm), ('ijkl,ji->kl','ijkl,li->kj'))
    eri1 = mol.intor('int2e_ip1_sph', comp=3).reshape([3]+[nao]*4)
    j1 = get_jk(mol, dm, 'ijkl,lk->ij', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    print(numpy.allclose(j1, numpy.einsum('xijkl,lk->xij', eri1, dm)))

    mol1 = gto.M(atom='He 2 0 0', basis='6-31g')
    nao1 = mol1.nao_nr()
    dm1 = numpy.random.random((nao1,nao1))
    eri0 = gto.conc_mol(mol, mol1).intor('int2e_sph').reshape([nao+nao1]*4)
    jcross = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', aosym='s4')
    ecoul = numpy.einsum('ij,ij', jcross, dm1)
    print(numpy.allclose(jcross, numpy.einsum('ijkl,lk->ij', eri0[nao:,nao:,:nao,:nao], dm)))
    print(ecoul-numpy.einsum('ijkl,lk,ij', eri0[nao:,nao:,:nao,:nao], dm, dm1))
    kcross = get_jk((mol1,mol,mol,mol1), dm, scripts='ijkl,jk->il')
    ex = numpy.einsum('ij,ji', kcross, dm1)
    print(numpy.allclose(kcross, numpy.einsum('ijkl,jk->il', eri0[nao:,:nao,:nao,nao:], dm)))
    print(ex-numpy.einsum('ijkl,jk,li', eri0[nao:,:nao,:nao,nao:], dm, dm1))
    kcross = get_jk((mol1,mol,mol,mol1), dm1, scripts='ijkl,li->kj')
    ex = numpy.einsum('ij,ji', kcross, dm)
    print(numpy.allclose(kcross, numpy.einsum('ijkl,li->kj', eri0[nao:,:nao,:nao,nao:], dm1)))
    print(ex-numpy.einsum('ijkl,jk,li', eri0[nao:,:nao,:nao,nao:], dm, dm1))

    j1part = get_jk((mol1,mol1,mol,mol), dm1[:1,:1], scripts='ijkl,ji->kl', intor='int2e',
                    shls_slice=(0,1,0,1,0,mol.nbas,0,mol.nbas))
    print(numpy.allclose(j1part, numpy.einsum('ijkl,ji->kl', eri0[nao:nao+1,nao:nao+1,:nao,:nao], dm1[:1,:1])))
    k1part = get_jk((mol1,mol,mol,mol1), dm1[:,:1], scripts='ijkl,li->kj', intor='int2e',
                    shls_slice=(0,1,0,1,0,mol.nbas,0,mol1.nbas))
    print(numpy.allclose(k1part, numpy.einsum('ijkl,li->kj', eri0[nao:nao+1,:1,:nao,nao:], dm1[:,:1])))

    j1part = get_jk(mol, dm[:1,1:2], scripts='ijkl,ji->kl', intor='int2e',
                    shls_slice=(1,2,0,1,0,mol.nbas,0,mol.nbas))
    print(numpy.allclose(j1part, numpy.einsum('ijkl,ji->kl', eri0[1:2,:1,:nao,:nao], dm[:1,1:2])))
    k1part = get_jk(mol, dm[:,1:2], scripts='ijkl,li->kj', intor='int2e',
                    shls_slice=(1,2,0,1,0,mol.nbas,0,mol.nbas))
    print(numpy.allclose(k1part, numpy.einsum('ijkl,li->kj', eri0[:1,1:2,:nao,:nao], dm[:,1:2])))

    eri1 = gto.conc_mol(mol, mol1).intor('int2e_ip1_sph',comp=3).reshape([3]+[nao+nao1]*4)
    j1cross = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
    print(numpy.allclose(j1cross, numpy.einsum('xijkl,lk->xij', eri1[:,nao:,nao:,:nao,:nao], dm)))
    j1part = get_jk((mol1,mol1,mol,mol), dm, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3,
                    shls_slice=(0,1,0,1,0,mol.nbas,0,mol.nbas))
    print(numpy.allclose(j1part, numpy.einsum('xijkl,lk->xij', eri1[:,nao:nao+1,nao:nao+1,:nao,:nao], dm)))
