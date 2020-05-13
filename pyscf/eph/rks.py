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
# Author: Yang Gao <younggao1994@gmail.com>

#
'''
Analytical electron-phonon matrix for restricted kohm sham
'''

import numpy as np
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint
from pyscf.eph import rhf as rhf_eph
from pyscf import lib

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    """" This functions is slightly different from hessian.rks._get_vxc_deriv1 in that <\nabla u|Vxc|v> is removed"""
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    vmat = np.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                rho1 = np.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                aow = np.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        v_ip = np.zeros((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            #rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)
            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = rks_hess._make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = numint._rks_gga_wv1(rho, dR_rho1[0], vxc, fxc, weight)
                wv[1] = numint._rks_gga_wv1(rho, dR_rho1[1], vxc, fxc, weight)
                wv[2] = numint._rks_gga_wv1(rho, dR_rho1[2], vxc, fxc, weight)
                aow = np.einsum('npi,Xnp->Xpi', ao[:4], wv)
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmat

def get_eph(ephobj, mo1, omega, vec, mo_rep):
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1 = dict([(int(k), mo1[k]) for k in mo1])

    mol = ephobj.mol
    mf = ephobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omg, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    vnuc_deriv = ephobj.vnuc_generator(mol)
    aoslices = mol.aoslice_by_atom()
    vind = rhf_eph.rhf_deriv_generator(mf, mf.mo_coeff, mf.mo_occ)
    mocc = mf.mo_coeff[:,mf.mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    natoms = mol.natm
    nao = mol.nao_nr()
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    vxc1ao = _get_vxc_deriv1(ephobj, mf.mo_coeff, mf.mo_occ, max_memory)
    vcore = []
    for ia in range(natoms):
        h1 = vnuc_deriv(ia)
        v1 = vind(mo1[ia])
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3

        if abs(hyb)>1e-10:
            vj1, vk1 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1], #vj1
                                      'li->s1kj', -dm0[:,p0:p1]], #vk1
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            if abs(omg) > 1e-10:
                with mol.with_range_coulomb(omg):
                    vk1 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1]], # vk1
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
        else:
            vj1 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1]], # vj1
                                        shls_slice=shls_slice)
            veff = vj1[0]
        vtot = h1 + v1 + veff + vxc1ao[ia] + veff.transpose(0,2,1)
        vcore.append(vtot)
    vcore = np.asarray(vcore).reshape(-1,nao,nao)
    mass = mol.atom_mass_list() * 1836.15
    nmodes, natoms = len(omega), len(mass)
    vec = vec.reshape(natoms, 3, nmodes)
    for i in range(natoms):
        for j in range(nmodes):
            vec[i,:,j] /= np.sqrt(2*mass[i]*omega[j])
    vec = vec.reshape(3*natoms,nmodes)
    mat = np.einsum('xJ,xuv->Juv', vec, vcore, optimize=True)
    if mo_rep:
        mat = np.einsum('Juv,up,vq->Jpq', mat, mf.mo_coeff.conj(), mf.mo_coeff, optimize=True)
    return mat


class EPH(rks_hess.Hessian):
    def __init__(self, scf_method):
        rks_hess.Hessian.__init__(self, scf_method)
        self.CUTOFF_FREQUENCY=80

    get_mode = rhf_eph.get_mode
    get_eph = get_eph
    vnuc_generator = rhf_eph.vnuc_generator
    kernel = rhf_eph.kernel

if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.M()
    mol.atom = [['O', [0.000000000000, 0.000000002577,0.868557119905]],
                ['H', [0.000000000000,-1.456050381698,2.152719488376]],
                ['H', [0.000000000000, 1.456050379121,2.152719486067]]]

    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build() # this is a pre-computed relaxed geometry

    mf = dft.RKS(mol)
    mf.grids.level=6
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)


    myeph = EPH(mf)
    eph, omega = myeph.kernel(mo_rep=True)
    print(np.amax(eph))
