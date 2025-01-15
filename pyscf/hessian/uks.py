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
Non-relativistic UKS analytical Hessian
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import uhf as uhf_hess
from pyscf.hessian import rks as rks_hess
from pyscf.dft import numint
_get_jk = rhf_hess._get_jk


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    if mf.do_nlc():
        raise NotImplementedError('UKS Hessian for NLC functional')

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)

    # Energy weighted density matrix
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    dme0 = numpy.einsum('pi,qi,i->pq', mocca, mocca, mo_ea)
    dme0+= numpy.einsum('pi,qi,i->pq', moccb, moccb, mo_eb)

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    de2, ej, ek = uhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             with_k=hybrid)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veffa_diag, veffb_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if hybrid and omega != 0:
        with mol.with_range_coulomb(omega):
            vk1a, vk1b = _get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                                 ['jk->s1il', dm0a, 'jk->s1il', dm0b])
        veffa_diag -= (alpha-hyb) * vk1a.reshape(3,3,nao,nao)
        veffb_diag -= (alpha-hyb) * vk1b.reshape(3,3,nao,nao)
    vk1a = vk1b = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    vxca, vxcb = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        veffa = vxca[ia]
        veffb = vxcb[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid and omega != 0:
            with mol.with_range_coulomb(omega):
                vk1a, vk1b, vk2a, vk2b = \
                        _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                ['li->s1kj', dm0a[:,p0:p1],
                                 'li->s1kj', dm0b[:,p0:p1],
                                 'lj->s1ki', dm0a         ,
                                 'lj->s1ki', dm0b         ],
                                shls_slice=shls_slice)
            veffa -= (alpha-hyb) * vk1a.reshape(3,3,nao,nao)
            veffb -= (alpha-hyb) * vk1b.reshape(3,3,nao,nao)
            veffa[:,:,:,p0:p1] -= (alpha-hyb) * vk2a.reshape(3,3,nao,p1-p0)
            veffb[:,:,:,p0:p1] -= (alpha-hyb) * vk2b.reshape(3,3,nao,p1-p0)
            t1 = log.timer_debug1('range-separated int2e_ip1ip2 for atom %d'%ia, *t1)
            with mol.with_range_coulomb(omega):
                vk1a, vk1b = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                     ['li->s1kj', dm0a[:,p0:p1],
                                      'li->s1kj', dm0b[:,p0:p1]],
                                     shls_slice=shls_slice)
            veffa -= (alpha-hyb) * vk1a.transpose(0,2,1).reshape(3,3,nao,nao)
            veffb -= (alpha-hyb) * vk1b.transpose(0,2,1).reshape(3,3,nao,nao)
            t1 = log.timer_debug1('range-separated int2e_ipvip1 for atom %d'%ia, *t1)
        vk1a = vk1b = vk2a = vk2b = None

        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veffa_diag[:,:,p0:p1], dm0a[p0:p1])*2
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veffb_diag[:,:,p0:p1], dm0b[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veffa[:,:,q0:q1], dm0a[q0:q1])*2
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veffb[:,:,q0:q1], dm0b[q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1aoa, h1aob = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid:
            vj1a, vj1b, vj2a, vj2b, vk1a, vk1b, vk2a, vk2b = \
                    _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                            ['ji->s2kl', -dm0a[:,p0:p1], 'ji->s2kl', -dm0b[:,p0:p1],
                             'lk->s1ij', -dm0a         , 'lk->s1ij', -dm0b         ,
                             'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                             'jk->s1il', -dm0a         , 'jk->s1il', -dm0b         ],
                            shls_slice=shls_slice)
            vj1 = vj1a + vj1b
            vj2 = vj2a + vj2b
            veffa = vj1 - hyb * vk1a
            veffb = vj1 - hyb * vk1b
            veffa[:,p0:p1] += vj2 - hyb * vk2a
            veffb[:,p0:p1] += vj2 - hyb * vk2b
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    vk1a, vk1b, vk2a, vk2b = \
                            _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                    ['li->s1kj', -dm0a[:,p0:p1],
                                     'li->s1kj', -dm0b[:,p0:p1],
                                     'jk->s1il', -dm0a         ,
                                     'jk->s1il', -dm0b         ],
                                    shls_slice=shls_slice)
                veffa -= (alpha-hyb) * vk1a
                veffb -= (alpha-hyb) * vk1b
                veffa[:,p0:p1] -= (alpha-hyb) * vk2a
                veffb[:,p0:p1] -= (alpha-hyb) * vk2b
        else:
            vj1a, vj1b, vj2a, vj2b = \
                    _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                            ['ji->s2kl', -dm0a[:,p0:p1], 'ji->s2kl', -dm0b[:,p0:p1],
                             'lk->s1ij', -dm0a         , 'lk->s1ij', -dm0b         ],
                            shls_slice=shls_slice)
            vj1 = vj1a + vj1b
            vj2 = vj2a + vj2b
            veffa = vj1
            veffb = vj1.copy()
            veffa[:,p0:p1] += vj2
            veffb[:,p0:p1] += vj2
        h1 = hcore_deriv(ia)
        h1aoa[ia] += h1 + veffa + veffa.transpose(0,2,1)
        h1aob[ia] += h1 + veffb + veffb.transpose(0,2,1)

    return h1aoa, h1aob

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
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
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmata = numpy.zeros((6,nao,nao))
    vmatb = numpy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 1, xctype=xctype)[1]
            wv = weight * vxc[:,0]
            aowa = numint._scale_ao(ao[0], wv[0])
            aowb = numint._scale_ao(ao[0], wv[1])
            for i in range(6):
                vmata[i] += numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
            aowa = aowb = None

    elif xctype == 'GGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 1, xctype=xctype)[1]
            wv = weight * vxc
            aowa = numint._scale_ao(ao[:4], wv[0,:4])
            aowb = numint._scale_ao(ao[:4], wv[1,:4])
            for i in range(6):
                vmata[i] += numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
            contract_(vmata[0], ao, [XXX,XXY,XXZ], wv[0], mask)
            contract_(vmata[1], ao, [XXY,XYY,XYZ], wv[0], mask)
            contract_(vmata[2], ao, [XXZ,XYZ,XZZ], wv[0], mask)
            contract_(vmata[3], ao, [XYY,YYY,YYZ], wv[0], mask)
            contract_(vmata[4], ao, [XYZ,YYZ,YZZ], wv[0], mask)
            contract_(vmata[5], ao, [XZZ,YZZ,ZZZ], wv[0], mask)
            contract_(vmatb[0], ao, [XXX,XXY,XXZ], wv[1], mask)
            contract_(vmatb[1], ao, [XXY,XYY,XYZ], wv[1], mask)
            contract_(vmatb[2], ao, [XXZ,XYZ,XZZ], wv[1], mask)
            contract_(vmatb[3], ao, [XYY,YYY,YYZ], wv[1], mask)
            contract_(vmatb[4], ao, [XYZ,YYZ,YZZ], wv[1], mask)
            contract_(vmatb[5], ao, [XZZ,YZZ,ZZZ], wv[1], mask)
            vxc = aowa = aowb = None

    elif xctype == 'MGGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:10], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:10], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,4] *= .5  # for the factor 1/2 in tau
            aowa = numint._scale_ao(ao[:4], wv[0,:4])
            aowb = numint._scale_ao(ao[:4], wv[1,:4])
            for i in range(6):
                vmata[i] += numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
            contract_(vmata[0], ao, [XXX,XXY,XXZ], wv[0], mask)
            contract_(vmata[1], ao, [XXY,XYY,XYZ], wv[0], mask)
            contract_(vmata[2], ao, [XXZ,XYZ,XZZ], wv[0], mask)
            contract_(vmata[3], ao, [XYY,YYY,YYZ], wv[0], mask)
            contract_(vmata[4], ao, [XYZ,YYZ,YZZ], wv[0], mask)
            contract_(vmata[5], ao, [XZZ,YZZ,ZZZ], wv[0], mask)
            contract_(vmatb[0], ao, [XXX,XXY,XXZ], wv[1], mask)
            contract_(vmatb[1], ao, [XXY,XYY,XYZ], wv[1], mask)
            contract_(vmatb[2], ao, [XXZ,XYZ,XZZ], wv[1], mask)
            contract_(vmatb[3], ao, [XYY,YYY,YYZ], wv[1], mask)
            contract_(vmatb[4], ao, [XYZ,YYZ,YZZ], wv[1], mask)
            contract_(vmatb[5], ao, [XZZ,YZZ,ZZZ], wv[1], mask)

            aowa = [numint._scale_ao(ao[i], wv[0,4]) for i in range(1, 4)]
            aowb = [numint._scale_ao(ao[i], wv[1,4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmata[i] += numint._dot_ao_ao(mol, ao[j], aowa[0], mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[j], aowb[0], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmata[i] += numint._dot_ao_ao(mol, ao[j], aowa[1], mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[j], aowb[1], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmata[i] += numint._dot_ao_ao(mol, ao[j], aowa[2], mask, shls_slice, ao_loc)
                vmatb[i] += numint._dot_ao_ao(mol, ao[j], aowb[2], mask, shls_slice, ao_loc)

    vmata = vmata[[0,1,2, 1,3,4, 2,4,5]].reshape(3,3,nao,nao)
    vmatb = vmatb[[0,1,2, 1,3,4, 2,4,5]].reshape(3,3,nao,nao)
    return vmata, vmatb

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
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

    vmata = numpy.zeros((mol.natm,3,3,nao,nao))
    vmatb = numpy.zeros((mol.natm,3,3,nao,nao))
    ipipa = numpy.zeros((3,3,nao,nao))
    ipipb = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc[:,0]
            aow = numpy.einsum('xpi,p->xpi', ao[1:4], wv[0])
            rks_hess._d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = numpy.einsum('xpi,p->xpi', ao[1:4], wv[1])
            rks_hess._d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            wf = weight * fxc[:,0,:,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1a = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0a[:,p0:p1]) * 2
                rho1b = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0b[:,p0:p1]) * 2
                wv  = wf[0,:,None] * rho1a
                wv += wf[1,:,None] * rho1b
                # aow ~ rho1 ~ d/dR1
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[0])
                rks_hess._d1d2_dot_(vmata[ia], mol, ao[1:4], aow, mask, ao_loc, False)
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[1])
                rks_hess._d1d2_dot_(vmatb[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0a = ao_dm0b = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,:,:,p0:p1] += ipipa[:,:,:,p0:p1]
            vmatb[ia,:,:,:,p0:p1] += ipipb[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            aow = rks_grad._make_dR_dao_w(ao, wv[0])
            rks_hess._d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = rks_grad._make_dR_dao_w(ao, wv[1])
            rks_hess._d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wva, wvb = wv

                aow = rks_grad._make_dR_dao_w(ao, wva[0])
                rks_grad._d1_dot_(vmata[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wva[1])
                rks_grad._d1_dot_(vmata[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wva[2])
                rks_grad._d1_dot_(vmata[ia,2], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmata[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                aow = rks_grad._make_dR_dao_w(ao, wvb[0])
                rks_grad._d1_dot_(vmatb[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wvb[1])
                rks_grad._d1_dot_(vmatb[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wvb[2])
                rks_grad._d1_dot_(vmatb[ia,2], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmatb[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0a = ao_dm0b = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,:,:,p0:p1] += ipipa[:,:,:,p0:p1]
            vmata[ia,:,:,:,p0:p1] += ipipa[:,:,p0:p1].transpose(1,0,3,2)
            vmatb[ia,:,:,:,p0:p1] += ipipb[:,:,:,p0:p1]
            vmatb[ia,:,:,:,p0:p1] += ipipb[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:10], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:10], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv[0])
            rks_hess._d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = rks_grad._make_dR_dao_w(ao, wv[1])
            rks_hess._d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[0,4]) for i in range(4, 10)]
            rks_hess._d1d2_dot_(ipipa, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            rks_hess._d1d2_dot_(ipipa, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            rks_hess._d1d2_dot_(ipipa, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)
            aow = [numint._scale_ao(ao[i], wv[1,4]) for i in range(4, 10)]
            rks_hess._d1d2_dot_(ipipb, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            rks_hess._d1d2_dot_(ipipb, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            rks_hess._d1d2_dot_(ipipb, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wv[:,:,4] *= .5
                wva, wvb = wv

                aow = rks_grad._make_dR_dao_w(ao, wva[0])
                rks_grad._d1_dot_(vmata[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wva[1])
                rks_grad._d1_dot_(vmata[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wva[2])
                rks_grad._d1_dot_(vmata[ia,2], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmata[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                aow = rks_grad._make_dR_dao_w(ao, wvb[0])
                rks_grad._d1_dot_(vmatb[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wvb[1])
                rks_grad._d1_dot_(vmatb[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wvb[2])
                rks_grad._d1_dot_(vmatb[ia,2], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmatb[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                aow = [numint._scale_ao(ao[1], wva[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmata[ia], mol, [ao[XX], ao[XY], ao[XZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[2], wva[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmata[ia], mol, [ao[YX], ao[YY], ao[YZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[3], wva[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmata[ia], mol, [ao[ZX], ao[ZY], ao[ZZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[1], wvb[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmatb[ia], mol, [ao[XX], ao[XY], ao[XZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[2], wvb[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmatb[ia], mol, [ao[YX], ao[YY], ao[YZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[3], wvb[i,4]) for i in range(3)]
                rks_hess._d1d2_dot_(vmatb[ia], mol, [ao[ZX], ao[ZY], ao[ZZ]], aow, mask, ao_loc, False)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,:,:,p0:p1] += ipipa[:,:,:,p0:p1]
            vmata[ia,:,:,:,p0:p1] += ipipa[:,:,p0:p1].transpose(1,0,3,2)
            vmatb[ia,:,:,:,p0:p1] += ipipb[:,:,:,p0:p1]
            vmatb[ia,:,:,:,p0:p1] += ipipb[:,:,p0:p1].transpose(1,0,3,2)

    return vmata, vmatb

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

    vmata = numpy.zeros((mol.natm,3,nao,nao))
    vmatb = numpy.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-(vmata.size+vmatb.size)*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc[:,0]
            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            aow1a = numpy.einsum('xpi,p->xpi', ao[1:], wv[0])
            aow1b = numpy.einsum('xpi,p->xpi', ao[1:], wv[1])
            wf = weight * fxc[:,0,:,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1a = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0a[:,p0:p1])
                rho1b = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0b[:,p0:p1])
                wv  = wf[0,:,None] * rho1a
                wv += wf[1,:,None] * rho1b
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[0])
                aow[:,:,p0:p1] += aow1a[:,:,p0:p1]
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[1])
                aow[:,:,p0:p1] += aow1b[:,:,p0:p1]
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = aow1a = aow1b = None

        for ia in range(mol.natm):
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        vipa = numpy.zeros((3,nao,nao))
        vipb = numpy.zeros((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            rks_grad._gga_grad_sum_(vipa, mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vipb, mol, ao, wv[1], mask, ao_loc)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wva, wvb = wv

                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,p0:p1] += vipa[:,p0:p1]
            vmatb[ia,:,p0:p1] += vipb[:,p0:p1]
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        rks_hess._check_mgga_grids(grids)
        ao_deriv = 2
        vipa = numpy.zeros((3,nao,nao))
        vipb = numpy.zeros((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa = ni.eval_rho2(mol, ao[:10], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:10], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5
            rks_grad._gga_grad_sum_(vipa, mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vipb, mol, ao, wv[1], mask, ao_loc)

            rks_grad._tau_grad_dot_(vipa, mol, ao, wv[0,4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vipb, mol, ao, wv[1,4], mask, ao_loc, True)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wv[:,:,4] *= .25
                wva, wvb = wv

                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)

                for j in range(1, 4):
                    aow = [numint._scale_ao(ao[j], wva[i,4]) for i in range(3)]
                    rks_grad._d1_dot_(vmata[ia], mol, aow, ao[j], mask, ao_loc, True)
                    aow = [numint._scale_ao(ao[j], wvb[i,4]) for i in range(3)]
                    rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[j], mask, ao_loc, True)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,p0:p1] += vipa[:,p0:p1]
            vmatb[ia,:,p0:p1] += vipb[:,p0:p1]
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    return vmata, vmatb


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic UKS hessian'''

    _keys = {'grids', 'grid_response'}

    def __init__(self, mf):
        uhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    hess_elec = uhf_hess.hess_elec
    gen_hop = uhf_hess.gen_hop
    solve_mo1 = uhf_hess.Hessian.solve_mo1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1

from pyscf import dft
dft.uks.UKS.Hessian = dft.uks_symm.UKS.Hessian = lib.class_as_method(Hessian)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    #xc_code = 'lda,vwn'
    xc_code = 'wb97x'
    #xc_code = 'b3lyp'

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = dft.UKS(mol)
    mf.xc = xc_code
    mf.conv_tol = 1e-14
    mf.kernel()
    n3 = mol.natm * 3
    hobj = mf.Hessian()
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.fp(e2) - -0.42286407986042956)
    print(lib.fp(e2) - -0.45453541215680582)
    print(lib.fp(e2) - -0.41385170026016327)

    mol.spin = 2
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-14
    mf.xc = xc_code
    mf.scf()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            mf = dft.UKS(mol).run(conv_tol=1e-14, xc=xc_code).run()
            e1a = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = dft.UKS(mol).run(conv_tol=1e-14, xc=xc_code).run()
            e1b = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-4))

# \partial^2 E / \partial R \partial R'
    e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    e2 += hobj.hess_nuc(mol)
    e2 = e2.transpose(0,2,1,3).reshape(n3,n3)
    def grad_partial_R(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_partial_R(ia, .5e-4) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-6))
