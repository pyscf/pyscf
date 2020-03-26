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
Analytical electron-phonon matrix for unrestricted kohn sham
'''
import numpy as np
from pyscf.hessian import uks as uks_hess
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint
from pyscf.eph import rhf as rhf_eph
from pyscf.eph.uhf import uhf_deriv_generator
import time
from pyscf import lib

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff[0].shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0a, dm0b = mf.make_rdm1(mo_coeff, mo_occ)

    vmata = np.zeros((mol.natm,3,nao,nao))
    vmatb = np.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-(vmata.size+vmatb.size)*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, (rhoa,rhob), 1, deriv=2)[1:3]
            vrho = vxc[0]
            u_u, u_d, d_d = fxc[0].T

            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1a = np.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0a[:,p0:p1])
                rho1b = np.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0b[:,p0:p1])

                wv = u_u * rho1a + u_d * rho1b
                wv *= weight
                aow = np.einsum('pi,xp->xpi', ao[0], wv)
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)

                wv = u_d * rho1a + d_d * rho1b
                wv *= weight
                aow = np.einsum('pi,xp->xpi', ao[0], wv)
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = aow1a = aow1b = None

        for ia in range(mol.natm):
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, (rhoa,rhob), 1, deriv=2)[1:3]

            wva, wvb = numint._uks_gga_wv0((rhoa,rhob), vxc, weight)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc)
                       for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc)
                       for i in range(4)]
            for ia in range(mol.natm):
                wva = dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices)
                wvb = dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices)
                wva[0], wvb[0] = numint._uks_gga_wv1((rhoa,rhob), (dR_rho1a[0],dR_rho1b[0]), vxc, fxc, weight)
                wva[1], wvb[1] = numint._uks_gga_wv1((rhoa,rhob), (dR_rho1a[1],dR_rho1b[1]), vxc, fxc, weight)
                wva[2], wvb[2] = numint._uks_gga_wv1((rhoa,rhob), (dR_rho1a[2],dR_rho1b[2]), vxc, fxc, weight)

                aow = np.einsum('npi,Xnp->Xpi', ao[:4], wva)
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = np.einsum('npi,Xnp->Xpi', ao[:4], wvb)
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = None

        for ia in range(mol.natm):
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmata, vmatb

def get_eph(ephobj, mo1, omega, vec, mo_rep):
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1a = mo1['0']
        mo1b = mo1['1']
        mo1a = dict([(int(k), mo1a[k]) for k in mo1a])
        mo1b = dict([(int(k), mo1b[k]) for k in mo1b])

    mol = ephobj.mol
    mf = ephobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omg, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    vnuc_deriv = ephobj.vnuc_generator(mol)
    aoslices = mol.aoslice_by_atom()

    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    vind = uhf_deriv_generator(mf, mf.mo_coeff, mf.mo_occ)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    vxc1aoa, vxc1aob = _get_vxc_deriv1(ephobj, mf.mo_coeff, mf.mo_occ, max_memory)
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = np.dot(mocca, mocca.T)
    dm0b = np.dot(moccb, moccb.T)

    natoms = mol.natm
    vcorea = []
    vcoreb = []

    for ia in range(natoms):
        h1 = vnuc_deriv(ia)
        moia = np.hstack((mo1a[ia], mo1b[ia]))
        v1 = vind(moia)
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb)>1e-10:
            vja, vjb, vka, vkb = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0a[:,p0:p1], #vja
                                      'ji->s2kl', -dm0b[:,p0:p1], #vjb
                                      'li->s1kj', -dm0a[:,p0:p1],
                                      'li->s1kj', -dm0b[:,p0:p1]], #vka
                                     shls_slice=shls_slice)
            vhfa = vja + vjb - hyb * vka
            vhfb = vjb + vja - hyb * vkb
            if abs(omg) > 1e-10:
                with mol.with_range_coulomb(omg):
                    vka, vkb = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0a[:,p0:p1],
                                          'li->s1kj', -dm0b[:,p0:p1]], # vk1
                                         shls_slice=shls_slice)
                vhfa -= (alpha-hyb) * vka
                vhfb -= (alpha-hyb) * vkb
        else:
            vja, vjb = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0a[:,p0:p1],
                                         'ji->s2kl', -dm0b[:,p0:p1]], # vj1
                                        shls_slice=shls_slice)
            vhfa = vhfb = vja + vjb
        vtota = h1 + v1[0] + vxc1aoa[ia] + vhfa + vhfa.transpose(0,2,1)
        vtotb = h1 + v1[1] + vxc1aob[ia] + vhfb + vhfb.transpose(0,2,1)
        vcorea.append(vtota)
        vcoreb.append(vtotb)

    vcorea = np.asarray(vcorea).reshape(-1,nao,nao)
    vcoreb = np.asarray(vcoreb).reshape(-1,nao,nao)

    mass = mol.atom_mass_list() * 1836.15
    nmodes, natoms = len(omega), len(mass)
    vec = vec.reshape(natoms, 3, nmodes)
    for i in range(natoms):
        for j in range(nmodes):
            vec[i,:,j] /= np.sqrt(2*mass[i]*omega[j])
    vec = vec.reshape(3*natoms,nmodes)
    mata = np.einsum('xJ,xuv->Juv', vec, vcorea)
    matb = np.einsum('xJ,xuv->Juv', vec, vcoreb)
    if mo_rep:
        mata = np.einsum('Juv,up,vq->Jpq', mata, mf.mo_coeff[0].conj(), mf.mo_coeff[0], optimize=True)
        matb = np.einsum('Juv,up,vq->Jpq', matb, mf.mo_coeff[1].conj(), mf.mo_coeff[1], optimize=True)
    return np.asarray([mata,matb])


class EPH(uks_hess.Hessian):
    def __init__(self, scf_method):
        uks_hess.Hessian.__init__(self, scf_method)
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

    mf = dft.UKS(mol)
    mf.grids.level=6
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)


    myeph = EPH(mf)
    (epha, ephb), omega = myeph.kernel()

    from pyscf.eph.rks import EPH as REPH
    mf = dft.RKS(mol)
    mf.grids.level=6
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    myeph = REPH(mf)
    rmat, omega = myeph.kernel()
    print(np.linalg.norm(epha-ephb))
    for i in range(len(rmat)):
        print(min(np.linalg.norm(epha[i]-rmat[i]), np.linalg.norm(epha[i]+rmat[i])))
