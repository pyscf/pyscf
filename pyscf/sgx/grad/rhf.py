#!/usr/bin/env python
# Copyright 2018-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#


import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df.incore import aux_e2
from pyscf.gto import moleintor
from pyscf.scf import _vhf
from pyscf.dft import gen_grid
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf import __config__
from pyscf.sgx.sgx_jk import _gen_jk_direct
from pyscf.sgx.sgx_jk import grids_response_cc as grids_response_sgx


def get_jk_favorj(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    if with_j:
        raise NotImplementedError('Gradient only available for SGX-K')
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    grids.build()
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt, sgx.pjs, tot_grids=grids.weights.size)

    if sgx.debug:
        raise NotImplemented #batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk_grad = _gen_jk_direct(mol, 's1', with_j, with_k, direct_scf_tol,
                                       None, sgx.pjs, tot_grids=grids.weights.size,
                                       grad=True)

    de = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)

    ovlp = mol.intor_symmetric('int1e_ovlp')
    proj = scipy.linalg.solve(sn, ovlp)
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)

    t1 = logger.timer_debug1(mol, "sgX initialziation", *t0)
    vj = numpy.zeros_like(dms)
    dvj = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    vk = numpy.zeros_like(dms)
    dvk = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    tnuc = 0, 0
    xed = numpy.zeros((nset, grids.weights.size)) # works if grids are not screened initially

    #for i0, i1 in lib.prange(0, ngrids, blksize):
    for ia, (coord_ia, weight_ia, weight1_ia) in enumerate(grids_response_sgx(grids)):
        ngrids = weight_ia.size
        dvk_tmp1 = numpy.zeros_like(dvk)
        dvk_tmp2 = numpy.zeros_like(dvk)
        for i0, i1 in lib.prange(0, ngrids, blksize):
            weights1 = weight1_ia[...,i0:i1]
            weights = weight_ia[i0:i1,None]
            coords = coord_ia[i0:i1]
            #ao = mol.eval_gto('GTOval', coords)
            if mol.cart:
                _ao = mol.eval_gto('GTOval_cart_deriv1', coords)
            else:
                _ao = mol.eval_gto('GTOval_sph_deriv1', coords)
            ao = _ao[0]
            dao = _ao[1:4]
            wao = ao * weights

            fg = lib.einsum('gi,xij->xgj', wao, proj_dm)
            mask = numpy.zeros(i1-i0, dtype=bool)
            for i in range(nset):
                mask |= numpy.any(fg[i]>gthrd, axis=1)
                mask |= numpy.any(fg[i]<-gthrd, axis=1)
            if False:#not numpy.all(mask):
                ao = ao[mask]
                dao = dao[:,mask]
                fg = fg[:,mask]
                coords = coords[mask]
                weights = weights[mask]
                weights1 = weights1[...,mask]

            if with_j:
                rhog = numpy.einsum('xgu,gu->xg', fg, ao)
            else:
                rhog = None

            if sgx.debug:
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                gbn = batch_nuc(mol, coords)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
                if with_j:
                    jpart = numpy.einsum('guv,xg->xuv', gbn, rhog)
                if with_k:
                    gv = lib.einsum('gtv,xgt->xgv', gbn, fg)
                gbn = None
            else:
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                if with_j: rhog = rhog.copy()
                jpart, gv = batch_jk(mol, coords, rhog, fg.copy(), weights)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                if with_j: rhog = rhog.copy()
                _, dgv = batch_jk_grad(mol, coords, rhog, fg.copy(), weights)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

            if with_j:
                vj += jpart
            if with_k:
                for i in range(nset):
                    vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
                    # THIS IS THE ORBITAL RESPONE TERM FOR SGX
                    dvk_tmp1[i] -= 1.0 * lib.einsum('xgu,gv->xuv', dao, gv[i]) # TODO factor of 2?
                    #dvk_tmp2[i] -= 1.0 * lib.einsum('xgu,gv->xuv', dgv[i], ao)
                    # TODO numerical stability? indexing?
                    xed = lib.einsum(
                        'gu,gu->g',
                        fg[i]/(weights + 1e-200),
                        gv[i]/(weights + 1e-200),
                    )
                    #de[:,:] += numpy.dot(weights1, xed).T

            jpart = gv = None
        dvk += dvk_tmp1 + dvk_tmp2
        for i in range(nset):
            #de[ia] -= (dvk_tmp1[i] * dms[i]).sum(axis=(1,2))
            #de[ia] -= numpy.einsum('xuv,uv->x', dvk_tmp1[i], dms[i])
            #de[ia] -= (0.5 * (dvk_tmp2[i] + dvk_tmp2[i].transpose(0,2,1)) * dms[i]).sum()
            pass

    for ia in range(mol.natm):
        print("DE i", de[ia])

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])

    #for i in range(nset):
    #    lib.hermi_triu(vj[i], inplace=True)
    vj, vk = dvj, dvk
    #if with_k and hermi == 1:
    #    vk = (vk + vk.transpose(0,1,3,2))*.5
    logger.timer(mol, "vj and vk", *t0)
    dm_shape = (nset, 3) + dms.shape[1:]
    vk = vk.reshape(dm_shape)
    vk = lib.tag_array(vk, de_grids=de)
    print("VK", vk.sum())
    return vj.reshape(dm_shape), vk


class Gradients(dfrhf_grad.Gradients):
    '''Restricted SGX Hartree-Fock gradients'''
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.sgx_grid_response = True
        dfrhf_grad.Gradients.__init__(self, mf)

    def get_j(self, mol=None, dm=None, hermi=0,
              direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)):
        return self.get_jk(mol, dm, with_k=False)[0]

    def get_k(self, mol=None, dm=None, hermi=0,
              direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)):
        return self.get_jk(mol, dm, with_j=False)[1]

    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj - vk*.5
        e1_apg = numpy.zeros(((mol or self.mol).natm, 3))
        if self.auxbasis_response:
            e1_apg += vj.aux
            logger.debug1(self, 'sum(auxbasis response) %s', e1_apg.sum(axis=0))
        if self.sgx_grid_response:
            e1_apg -= 0.5 * vk.de_grids # TODO factor here?
        vhf = lib.tag_array(vhf, aux=e1_apg)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response or self.sgx_grid_response:
            print('ADDING AUX')
            return envs['vhf'].aux[atom_id]
        else:
            return 0

    def get_jk(self, mol=None, dm=None, hermi=1, vhfopt=None, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is not None:
            raise NotImplementedError('No range-separated nuclear gradients')

        dm = self.base.make_rdm1()
        if with_j and self.base.with_df.dfj:
            print("DFJ")
            vj = super(Gradients, self).get_jk(self.mol, dm, hermi, with_j=True, with_k=False)[0]
            if with_k:
                vk = get_jk_favorj(self.base.with_df, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        elif with_j:
            raise NotImplementedError
        else:
            print("NO DFJ")
            vj, vk = get_jk_favorj(self.base.with_df, dm, hermi, with_j, with_k, direct_scf_tol)

        if with_k:
            vk = lib.tag_array(vk[0], de_grids=vk.de_grids)
        return vj, vk


Grad = Gradients
