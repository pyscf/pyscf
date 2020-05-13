#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#
# 
# #
# # Copyright 2019 Tencent America LLC. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # Author: Qiming Sun <osirpt.sun@gmail.com>
# #


import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.ao2mo.outcore import balance_partition
from pyscf.gto.moleintor import getints, make_cintopt
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad


def get_jk(mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True):
    if mol is None: mol = mf_grad.mol
    #if dm is None: dm = mf_grad.base.make_rdm1()
    #TODO: dm has to be the SCF density matrix in this version.  dm should be
    # extended to any 1-particle density matrix
    dm = mf_grad.base.make_rdm1()

    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    ao_loc = mol.ao_loc
    nbas = mol.nbas
    nauxbas = auxmol.nbas

    get_int3c_s1 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    get_int3c_s2 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')

    nao = mol.nao
    naux = auxmol.nao
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    auxslices = auxmol.aoslice_by_atom()
    aux_loc = auxmol.ao_loc
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = balance_partition(aux_loc, blksize)

    if not with_k:
        idx = numpy.arange(nao)
        dm_tril = dms + dms.transpose(0,2,1)
        dm_tril[:,idx,idx] *= .5
        dm_tril = lib.pack_tril(dm_tril)

        # (i,j|P)
        rhoj = numpy.empty((nset,naux))
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_s2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            rhoj[:,p0:p1] = numpy.einsum('wp,nw->np', int3c, dm_tril)
            int3c = None

        # (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1')
        rhoj = scipy.linalg.solve(int2c, rhoj.T, sym_pos=True).T
        int2c = None

        # (d/dX i,j|P)
        vj = numpy.zeros((nset,3,nao,nao))
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vj += numpy.einsum('xijp,np->nxij', int3c, rhoj[:,p0:p1])
            int3c = None

        if mf_grad.auxbasis_response:
            # (i,j|d/dX P)
            vjaux = numpy.empty((3,naux))
            for shl0, shl1, nL in ao_ranges:
                int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
                p0, p1 = aux_loc[shl0], aux_loc[shl1]
                vjaux[:,p0:p1] = numpy.einsum('xwp,mw,np->xp',
                                              int3c, dm_tril, rhoj[:,p0:p1])
                int3c = None

            # (d/dX P|Q)
            int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
            vjaux -= numpy.einsum('xpq,mp,nq->xp', int2c_e1, rhoj, rhoj)

            vjaux = [-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]]
            vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
        else:
            vj = -vj.reshape(out_shape)
        return vj, None

    mo_coeff = mf_grad.base.mo_coeff
    mo_occ = mf_grad.base.mo_occ
    nmo = mo_occ.shape[-1]
    if isinstance(mf_grad.base, scf.rohf.ROHF):
        mo_coeff = numpy.vstack((mo_coeff,mo_coeff))
        mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
        mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
        assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
        mo_occ = numpy.vstack((mo_occa, mo_occb))

    mo_coeff = numpy.asarray(mo_coeff).reshape(-1,nao,nmo)
    mo_occ   = numpy.asarray(mo_occ).reshape(-1,nmo)
    rhoj = numpy.zeros((nset,naux))
    f_rhok = lib.H5TmpFile()
    orbo = []
    for i in range(nset):
        c = numpy.einsum('pi,i->pi', mo_coeff[i][:,mo_occ[i]>0],
                         numpy.sqrt(mo_occ[i][mo_occ[i]>0]))
        nocc = c.shape[1]
        orbo.append(c)

    # (P|Q)
    int2c = scipy.linalg.cho_factor(auxmol.intor('int2c2e', aosym='s1'))

    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = max_memory * .5e6/8 / (naux*nao)
    mol_ao_ranges = balance_partition(ao_loc, blksize)
    nsteps = len(mol_ao_ranges)
    for istep, (shl0, shl1, nd) in enumerate(mol_ao_ranges):
        int3c = get_int3c_s1((0, nbas, shl0, shl1, 0, nauxbas))
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        rhoj += numpy.einsum('nlk,klp->np', dms[:,p0:p1], int3c)
        for i in range(nset):
            v = lib.einsum('ko,klp->plo', orbo[i], int3c)
            v = scipy.linalg.cho_solve(int2c, v.reshape(naux,-1))
            f_rhok['%s/%s'%(i,istep)] = v.reshape(naux,p1-p0,-1)
        int3c = v = None

    rhoj = scipy.linalg.cho_solve(int2c, rhoj.T).T
    int2c = None

    def load(set_id, p0, p1):
        nocc = orbo[set_id].shape[1]
        buf = numpy.empty((p1-p0,nocc,nao))
        col1 = 0
        for istep in range(nsteps):
            dat = f_rhok['%s/%s'%(set_id,istep)][p0:p1]
            col0, col1 = col1, col1 + dat.shape[1]
            buf[:p1-p0,:,col0:col1] = dat.transpose(0,2,1)
        return buf

    vj = numpy.zeros((nset,3,nao,nao))
    vk = numpy.zeros((nset,3,nao,nao))
    # (d/dX i,j|P)
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        vj += numpy.einsum('xijp,np->nxij', int3c, rhoj[:,p0:p1])
        for i in range(nset):
            tmp = lib.einsum('xijp,jo->xipo', int3c, orbo[i])
            rhok = load(i, p0, p1)
            vk[i] += lib.einsum('xipo,pok->xik', tmp, rhok)
            tmp = rhok = None
        int3c = None

    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao*nocc), 20), naux))
    rhok_oo = []
    for i in range(nset):
        nocc = orbo[i].shape[1]
        tmp = numpy.empty((naux,nocc,nocc))
        for p0, p1 in lib.prange(0, naux, blksize):
            rhok = load(i, p0, p1)
            tmp[p0:p1] = lib.einsum('pok,kr->por', rhok, orbo[i])
        rhok_oo.append(tmp)
        rhok = tmp = None

    if mf_grad.auxbasis_response:
        vjaux = numpy.zeros((3,naux))
        vkaux = numpy.zeros((3,naux))
        # (i,j|d/dX P)
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            int3c = int3c.transpose(0,2,1).reshape(3*(p1-p0),-1)
            int3c = lib.unpack_tril(int3c)
            int3c = int3c.reshape(3,p1-p0,nao,nao)
            vjaux[:,p0:p1] = numpy.einsum('xpij,mji,np->xp',
                                          int3c, dms, rhoj[:,p0:p1])
            for i in range(nset):
                tmp = rhok_oo[i][p0:p1]
                tmp = lib.einsum('por,ir->pio', tmp, orbo[i])
                tmp = lib.einsum('pio,jo->pij', tmp, orbo[i])
                vkaux[:,p0:p1] += lib.einsum('xpij,pij->xp', int3c, tmp)
        int3c = tmp = None

        # (d/dX P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1')
        vjaux -= numpy.einsum('xpq,mp,nq->xp', int2c_e1, rhoj, rhoj)
        for i in range(nset):
            tmp = lib.einsum('pij,qij->pq', rhok_oo[i], rhok_oo[i])
            vkaux -= numpy.einsum('xpq,pq->xp', int2c_e1, tmp)

        vjaux = [-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]]
        vkaux = [-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]]
        vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
        vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
    else:
        vj = -vj.reshape(out_shape)
        vk = -vk.reshape(out_shape)
    return vj, vk

def _int3c_wrapper(mol, auxmol, intor, aosym):
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix(intor)
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    def get_int3c(shls_slice=None):
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
        else:
            shls_slice = shls_slice[:4] + (nbas+shls_slice[4], nbas+shls_slice[5])
        return getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice,
                       aosym=aosym, cintopt=opt)
    return get_int3c


class Gradients(rhf_grad.Gradients):
    '''Restricted density-fitting Hartree-Fock gradients'''
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = True
        rhf_grad.Gradients.__init__(self, mf)

    get_jk = get_jk

    def get_j(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_k=False)[0]

    def get_k(self, mol=None, dm=None, hermi=0):
        return self.get_jk(mol, dm, with_j=False)[1]

    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj - vk*.5
        if self.auxbasis_response:
            e1_aux = vj.aux - vk.aux*.5
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vhf = lib.tag_array(vhf, aux=e1_aux)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis='ccpvdz-jkfit').run()
    g = Gradients(mf).set(auxbasis_response=not False).kernel()
    print(lib.finger(g) - 0.0055166381900824879)
    g = Gradients(mf).kernel()
    print(lib.finger(g) - 0.005516638190173352)
    print(abs(g-scf.RHF(mol).run().Gradients().kernel()).max())
# -0.0000000000    -0.0000000000    -0.0241140368
#  0.0000000000     0.0043935801     0.0120570184
#  0.0000000000    -0.0043935801     0.0120570184

    mfs = mf.as_scanner()
    e1 = mfs([['O' , (0. , 0.     , 0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    e2 = mfs([['O' , (0. , 0.     ,-0.001)],
              [1   , (0. , -0.757 , 0.587)],
              [1   , (0. , 0.757  , 0.587)] ])
    print((e1-e2)/0.002*lib.param.BOHR)
