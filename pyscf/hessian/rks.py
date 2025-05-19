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
Non-relativistic RKS analytical Hessian
'''


import numpy
import ctypes
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint, gen_grid


# import pyscf.grad.rks to activate nuc_grad_method method
from pyscf.grad import rks  # noqa


min_grid_blksize = 128*128
NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-8

libdft = lib.load_library('libdft')
contract = numpy.einsum


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    de2, ej, ek = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             with_k=hybrid)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if hybrid and omega != 0:
        with mol.with_range_coulomb(omega):
            vk1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                                   ['jk->s1il', dm0])[0]
        veff_diag -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
    vk1 = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    vxc = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        veff = vxc[ia]
        if hybrid and omega != 0:
            with mol.with_range_coulomb(omega):
                vk1, vk2 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                            ['li->s1kj', dm0[:,p0:p1],  # vk1
                                             'lj->s1ki', dm0         ], # vk2
                                            shls_slice=shls_slice)
            veff -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
            veff[:,:,:,p0:p1] -= (alpha-hyb)*.5 * vk2.reshape(3,3,nao,p1-p0)
            t1 = log.timer_debug1('range-separated int2e_ip1ip2 for atom %d'%ia, *t1)
            with mol.with_range_coulomb(omega):
                vk1 = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                       ['li->s1kj', dm0[:,p0:p1]], # vk1
                                       shls_slice=shls_slice)[0]
            veff -= (alpha-hyb)*.5 * vk1.transpose(0,2,1).reshape(3,3,nao,nao)
            t1 = log.timer_debug1('range-separated int2e_ipvip1 for atom %d'%ia, *t1)
            vk1 = vk2 = None

        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veff[:,:,q0:q1], dm0[q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    if mf.do_nlc():
        h1ao += _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)

    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            veff = vj1
            veff[:,p0:p1] += vj2

        h1ao[ia] += veff + veff.transpose(0,2,1)
        h1ao[ia] += hcore_deriv(ia)

    return h1ao

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

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
            aow = None

    elif xctype == 'GGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)
            rho = vxc = wv = aow = None

    elif xctype == 'MGGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[4] *= .5  # for the factor 1/2 in tau
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[0], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[1], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[2], mask, shls_slice, ao_loc)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[0]
    if xctype == 'GGA':
        rho1 = numpy.zeros((3,4,ngrids))
    elif xctype == 'MGGA':
        rho1 = numpy.zeros((3,5,ngrids))
        ao_dm0_x = ao_dm0[1][:,p0:p1]
        ao_dm0_y = ao_dm0[2][:,p0:p1]
        ao_dm0_z = ao_dm0[3][:,p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_x)
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_y)
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_z)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_x)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_y)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_z)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_x)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_y)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_z)
        rho1[:,4] *= .5
    else:
        raise RuntimeError

    ao_dm0_0 = ao_dm0[0][:,p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0_0)
    rho1[0,1]+= numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_0)
    rho1[0,2]+= numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_0)
    rho1[0,3]+= numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_0)
    rho1[1,1]+= numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_0)
    rho1[1,2]+= numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_0)
    rho1[1,3]+= numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_0)
    rho1[2,1]+= numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_0)
    rho1[2,2]+= numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_0)
    rho1[2,3]+= numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[1][:,p0:p1])
    rho1[:,2] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[2][:,p0:p1])
    rho1[:,3] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[3][:,p0:p1])

    # *2 for |mu> DM <d_X nu|
    return rho1 * 2

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
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

    vmat = numpy.zeros((mol.natm,3,3,nao,nao))
    ipip = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                # aow ~ rho1 ~ d/dR1
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
            _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                wv[:,4] *= .5  # for the factor 1/2 in tau
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                aow = [numint._scale_ao(ao[1], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[XX], ao[XY], ao[XZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[2], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[YX], ao[YY], ao[YZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[3], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[ZX], ao[ZY], ao[ZZ]], aow, mask, ao_loc, False)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    return vmat

def _get_enlc_deriv2_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc energy 2nd derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    de2 = numpy.empty([mol.natm, mol.natm, 3, 3])

    def get_nlc_de(grad_obj, dm):
        from pyscf.grad.rks import _initialize_grids, get_nlc_vxc_full_response
        mol = grad_obj.mol

        mf = grad_obj.base
        ni = mf._numint
        _, nlcgrids = _initialize_grids(grad_obj)

        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            xc = mf.nlc
        enlc, vnlc = get_nlc_vxc_full_response(
            ni, mol, nlcgrids, xc, dm,
            max_memory=max_memory, verbose=grad_obj.verbose)

        aoslices = mol.aoslice_by_atom()
        de = numpy.zeros((mol.natm,3))
        for i_atom in range(mol.natm):
            p0, p1 = aoslices[i_atom, 2:]
            de[i_atom] += numpy.einsum('xij,ij->x', vnlc[:,p0:p1], dm[p0:p1]) * 2

        assert enlc is not None
        de += enlc
        return de

    dx = 1e-3
    mol_copy = mol.copy()
    mol_copy.verbose = 0
    grad_obj = mf.Gradients()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            grad_obj.reset(mol_copy)
            de_p = get_nlc_de(grad_obj, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            grad_obj.reset(mol_copy)
            de_m = get_nlc_de(grad_obj, dm0)

            de2[i_atom, :, i_xyz, :] = (de_p - de_m) / (2 * dx)
    grad_obj.reset(mol)

    return de2

def get_d2mu_dr2(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d2mu_dr2 = numpy.empty([3, 3, nao, ngrids])
    d2mu_dr2[0,0,:,:] = ao[XX, :, :]
    d2mu_dr2[0,1,:,:] = ao[XY, :, :]
    d2mu_dr2[1,0,:,:] = ao[XY, :, :]
    d2mu_dr2[0,2,:,:] = ao[XZ, :, :]
    d2mu_dr2[2,0,:,:] = ao[XZ, :, :]
    d2mu_dr2[1,1,:,:] = ao[YY, :, :]
    d2mu_dr2[1,2,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,1,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,2,:,:] = ao[ZZ, :, :]
    return d2mu_dr2

def get_d3mu_dr3(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d3mu_dr3 = numpy.empty([3, 3, 3, nao, ngrids])
    d3mu_dr3[0,0,0,:,:] = ao[XXX,:,:]
    d3mu_dr3[0,0,1,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,1,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[1,0,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,0,2,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,2,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[2,0,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,1,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,0,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,1,0,:,:] = ao[XYY,:,:]
    d3mu_dr3[0,1,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,0,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,2,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,0,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,1,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,0,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,2,0,:,:] = ao[XZZ,:,:]
    d3mu_dr3[1,1,1,:,:] = ao[YYY,:,:]
    d3mu_dr3[1,1,2,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[2,1,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,1,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,1,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,2,:,:] = ao[ZZZ,:,:]

    return d3mu_dr3

def get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    d2rho_dAdr = numpy.zeros([natm, 3, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[:, p0:p1].T)
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], nu_dot_dm)
        nu_dot_dm = None
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], mu_dot_dm)
        mu_dot_dm = None
        dnudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[p0:p1, :])
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dnudr_dot_dm)
        dnudr_dot_dm = None
        dmudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[:, p0:p1].T)
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dmudr_dot_dm)
        dmudr_dot_dm = None
    return d2rho_dAdr

def get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None
        natm = len(atom_to_grid_index_map)

        d2rho_dAdr_grid_response = numpy.zeros([natm, 3, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # d2rho_dAdr_response  = numpy.einsum('dDig,jg,ij->dDg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # d2rho_dAdr_response += numpy.einsum('dDig,jg,ij->dDg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            # d2rho_dAdr_response += numpy.einsum('dig,Djg,ij->dDg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0)
            # d2rho_dAdr_response += numpy.einsum('dig,Djg,ij->dDg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            d2rho_dAdr_response  = contract('dDig,ig->dDg', d2mu_dr2[:, :, :, associated_grid_index], dm_dot_mu_and_nu)
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            d2rho_dAdr_response += contract('dig,Dig->dDg', dmu_dr[:, :, associated_grid_index], dm_dot_dmu_and_dnu)
            dm_dot_dmu_and_dnu = None

            d2rho_dAdr_grid_response[i_atom][:, :, associated_grid_index] = d2rho_dAdr_response
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        d2rho_dAdr_grid_response  = contract('dDig,ig->dDg', d2mu_dr2, dm_dot_mu_and_nu)
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        d2rho_dAdr_grid_response += contract('dig,Dig->dDg', dmu_dr, dm_dot_dmu_and_dnu)
        dm_dot_dmu_and_dnu = None

    return d2rho_dAdr_grid_response

def get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('Djg,Dg->jg', dmu_dr, drho_dr)

    drho_dA = numpy.zeros([natm, 3, ngrids])
    dgamma_dA = numpy.zeros([natm, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]

        # drho_dA[i_atom, :, :] += numpy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[p0:p1, :])
        # drho_dA[i_atom, :, :] += numpy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], mu_dot_dm)

        # dgamma_dA[i_atom, :, :] += numpy.einsum('dDig,jg,Dg,ij->dg',
        #     -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dDig,jg,Dg,ij->dg',
        #     -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[:, p0:p1].T)
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dig,Djg,Dg,ij->dg',
        #     -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dig,Djg,Dg,ij->dg',
        #     -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[:, p0:p1].T)
        d2mudAdr_dot_drhodr = contract('dDig,Dg->dig', -d2mu_dr2[:, :, p0:p1, :], drho_dr)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, nu_dot_dm)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, mu_dot_dm)
        d2mudAdr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[p0:p1, :] @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[:, p0:p1].T @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
    dgamma_dA *= 2

    return drho_dA, dgamma_dA

def get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None

        natm = len(atom_to_grid_index_map)
        drho_dA_grid_response   = numpy.zeros([natm, 3, ngrids])
        dgamma_dA_grid_response = numpy.zeros([natm, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # rho_response  = numpy.einsum('dig,jg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # rho_response += numpy.einsum('dig,jg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            rho_response = contract('dig,ig->dg', dmu_dr[:, :, associated_grid_index], dm_dot_mu_and_nu)
            drho_dA_grid_response[i_atom][:, associated_grid_index] = rho_response
            rho_response = None

            # gamma_response  = numpy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index],
            #     mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += numpy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index],
            #     mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            # gamma_response += numpy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index],
            #     dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += numpy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index],
            #     dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            d2mudr2_dot_drhodr = contract('dDig,Dg->dig',
                d2mu_dr2[:, :, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response  = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
            d2mudr2_dot_drhodr = None
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig',
                dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            dmudr_dot_drhodr = contract('dig,dg->ig',
                dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
            dmudr_dot_drhodr = None
            dm_dot_dmu_and_dnu = None
            dgamma_dA_grid_response[i_atom][:, associated_grid_index] = gamma_response
            gamma_response = None
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        drho_dA_grid_response = contract('dig,ig->dg', dmu_dr, dm_dot_mu_and_nu)

        d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2, drho_dr)
        dgamma_dA_grid_response = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
        d2mudr2_dot_drhodr = None
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, drho_dr)
        dgamma_dA_grid_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
        dmudr_dot_drhodr = None
        dm_dot_dmu_and_dnu = None

    dgamma_dA_grid_response *= 2

    return drho_dA_grid_response, dgamma_dA_grid_response

def get_d2rhodAdB_d2gammadAdB(mol, grids_coords, dm0):
    """
        This function should never be used in practice. It requires crazy amount of memory,
        and it's left for debug purpose only. Use the contract function instead.
    """
    natm = mol.natm
    ngrids = grids_coords.shape[0]

    ao = numint.eval_ao(mol, grids_coords, deriv = 3)
    rho_drho = numint.eval_rho(mol, ao[:4, :], dm0, xctype = "GGA", hermi = 1, with_lapl = False)
    ao = ao.transpose(0,2,1) # order: component, ao, grid
    drho = rho_drho[1:4, :]
    mu = ao[0, :, :]
    dmu_dr = ao[1:4, :, :]
    d2mu_dr2 = get_d2mu_dr2(ao)
    d3mu_dr3 = get_d3mu_dr3(ao)

    aoslices = mol.aoslice_by_atom()
    d2rho_dAdB = numpy.zeros([natm, natm, 3, 3, ngrids])
    d2gamma_dAdB = numpy.zeros([natm, natm, 3, 3, ngrids])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]
        d2rho_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[pi0:pi1, :])
        d2rho_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDPig,jg,Pg,ij->dDg',
            d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDPig,jg,Pg,ij->dDg',
            d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,Pjg,Pg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,Pjg,Pg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[:, pi0:pi1].T)
        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            d2rho_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2gamma_dAdB += numpy.einsum('AdPg,BDPg->ABdDg', d2rho_dAdr, d2rho_dAdr)
    d2gamma_dAdB *= 2
    return d2rho_dAdB, d2gamma_dAdB

def contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices, fw_rho, fw_gamma):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d3mu_dr3.shape == (3, 3, 3, nao, ngrids)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('djg,dg->jg', dmu_dr, drho_dr)

    d2e_rho_dAdB = numpy.zeros([natm, natm, 3, 3])
    d2e_gamma_dAdB = numpy.zeros([natm, natm, 3, 3])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]

        nu_dot_dm = dm0[pi0:pi1, :] @ mu
        d2rho_dA2  = contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, pi0:pi1].T @ mu
        d2rho_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu_dot_dm)
        d2e_rho_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2rho_dA2, fw_rho)
        d2rho_dA2 = None

        d3mudA2dr_dot_drhodr = contract('dDPig,Pg->dDig', d3mu_dr3[:, :, :, pi0:pi1, :], drho_dr)
        d2gamma_dA2  = contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, nu_dot_dm)
        d2gamma_dA2 += contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, mu_dot_dm)
        d3mudA2dr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[pi0:pi1, :] @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[:, pi0:pi1].T @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        d2e_gamma_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2gamma_dA2, fw_gamma)
        d2gamma_dA2 = None

        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            dnudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB  = contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dnudr_dot_dm)
            dmudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2rho_dAdB += contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dmudr_dot_dm)
            d2e_rho_dAdB[i_atom, j_atom, :, :] += contract('dDg,g->dD', d2rho_dAdB, fw_rho)
            d2rho_dAdB = None

            drhodr_dot_d2mudAdr = contract('dDig,Dg->dig', d2mu_dr2[:, :, pi0:pi1, :], drho_dr)
            d2gamma_dAdB  = contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dnudr_dot_dm)
            dnudr_dot_dm = None
            d2gamma_dAdB += contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dmudr_dot_dm)
            dmudr_dot_dm = None
            drhodr_dot_d2mudAdr = None
            d2gamma_dAdB = contract('dDg,g->dD', d2gamma_dAdB, fw_gamma)
            d2e_gamma_dAdB[i_atom, j_atom, :, :] += d2gamma_dAdB
            d2e_gamma_dAdB[j_atom, i_atom, :, :] += d2gamma_dAdB.T
            d2gamma_dAdB = None

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2e_gamma_dAdB += contract('AdPg,BDPg->ABdD', d2rho_dAdr, d2rho_dAdr * fw_gamma)

    return d2e_rho_dAdB + 2 * d2e_gamma_dAdB

def _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    mol = hessobj.mol
    mf = hessobj.base

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = 2 * mocc @ mocc.T

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = mf._numint.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    # ao = numint.eval_ao(mol, grids.coords, deriv = 3)
    # rho_drho = numint.eval_rho(mol, ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    ngrids_full = grids.coords.shape[0]
    rho_drho = numpy.empty([4, ngrids_full])

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    ao_nbytes_per_grid = ((4*2) * mol.nao + 4) * 8 # factor of 2 from the ao sorting inside numint.eval_ao()
    ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of CPU memory for NLC energy second derivative, "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids_full}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids_full, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids_full)
        split_grids_coords = grids.coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1)
        split_rho_drho = numint.eval_rho(mol, split_ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)
        rho_drho[:, g0:g1] = split_rho_drho

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = numpy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = numpy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = numpy.empty(ngrids)
    W_i = numpy.empty(ngrids)
    A_i = numpy.empty(ngrids)
    B_i = numpy.empty(ngrids)
    C_i = numpy.empty(ngrids)
    E_i = numpy.empty(ngrids)

    libdft.VXC_vv10nlc_hessian_eval_UWABCE(
        U_i.ctypes.data_as(ctypes.c_void_p),
        W_i.ctypes.data_as(ctypes.c_void_p),
        A_i.ctypes.data_as(ctypes.c_void_p),
        B_i.ctypes.data_as(ctypes.c_void_p),
        C_i.ctypes.data_as(ctypes.c_void_p),
        E_i.ctypes.data_as(ctypes.c_void_p),
        grids_coords.ctypes.data_as(ctypes.c_void_p),
        grids_weights.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        omega_i.ctypes.data_as(ctypes.c_void_p),
        kappa_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )

    domega_drho_i         = numpy.empty(ngrids)
    domega_dgamma_i       = numpy.empty(ngrids)
    d2omega_drho2_i       = numpy.empty(ngrids)
    d2omega_dgamma2_i     = numpy.empty(ngrids)
    d2omega_drho_dgamma_i = numpy.empty(ngrids)
    libdft.VXC_vv10nlc_hessian_eval_omega_derivative(
        domega_drho_i.ctypes.data_as(ctypes.c_void_p),
        domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        gamma_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i
    f_rho_i   =   f_rho_i * grids_weights
    f_gamma_i = f_gamma_i * grids_weights

    aoslices = mol.aoslice_by_atom()
    natm = mol.natm

    # ao = numint.eval_ao(mol, grids.coords, deriv = 3)
    # ao = ao.transpose(0,2,1) # order: component, ao, grid
    # ao_nonzero_rho = ao[:, :, rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)
    # d3mu_dr3 = get_d3mu_dr3(ao_nonzero_rho)

    # drho_dA, dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices)
    # d2e = contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices,
    #                                      f_rho_i, f_gamma_i)

    drho_dA   = numpy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = numpy.empty([natm, 3, ngrids], order = "C")
    d2e = numpy.zeros([natm, natm, 3, 3])

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    ao_nbytes_per_grid = ((20 + 1*2 + 3*2 + 9 + 27) * mol.nao + (3*2 + 9) * mol.natm) * 8
    ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of CPU memory for NLC energy second derivative, "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        split_grids_coords = grids_coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 3)
        split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

        mu = split_ao[0, :, :]
        dmu_dr = split_ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(split_ao)
        d3mu_dr3 = get_d3mu_dr3(split_ao)
        split_drho_dr = nabla_rho_i[:, g0:g1]

        split_drho_dA, split_dgamma_dA = \
            get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices)
        drho_dA  [:, :, g0:g1] = split_drho_dA
        dgamma_dA[:, :, g0:g1] = split_dgamma_dA

        split_fw_rho   = f_rho_i  [g0:g1]
        split_fw_gamma = f_gamma_i[g0:g1]
        d2e += contract_d2rhodAdB_d2gammadAdB(
            d3mu_dr3, d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices, split_fw_rho, split_fw_gamma)

        split_ao = None
        mu = None
        dmu_dr = None
        d2mu_dr2 = None
        d3mu_dr3 = None
        split_drho_dA = None
        split_dgamma_dA = None

    drho_dA   = numpy.ascontiguousarray(drho_dA)
    dgamma_dA = numpy.ascontiguousarray(dgamma_dA)
    f_rho_A_i   = numpy.empty([mol.natm, 3, ngrids], order = "C")
    f_gamma_A_i = numpy.empty([mol.natm, 3, ngrids], order = "C")

    libdft.VXC_vv10nlc_hessian_eval_f_t(
        f_rho_A_i.ctypes.data_as(ctypes.c_void_p),
        f_gamma_A_i.ctypes.data_as(ctypes.c_void_p),
        grids_coords.ctypes.data_as(ctypes.c_void_p),
        grids_weights.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        omega_i.ctypes.data_as(ctypes.c_void_p),
        kappa_i.ctypes.data_as(ctypes.c_void_p),
        U_i.ctypes.data_as(ctypes.c_void_p),
        W_i.ctypes.data_as(ctypes.c_void_p),
        A_i.ctypes.data_as(ctypes.c_void_p),
        B_i.ctypes.data_as(ctypes.c_void_p),
        C_i.ctypes.data_as(ctypes.c_void_p),
        domega_drho_i.ctypes.data_as(ctypes.c_void_p),
        domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        dkappa_drho_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        d2kappa_drho2_i.ctypes.data_as(ctypes.c_void_p),
        drho_dA.ctypes.data_as(ctypes.c_void_p),
        dgamma_dA.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(3 * mol.natm),
    )

    d2e += contract("Adg,BDg->ABdD",   drho_dA,   f_rho_A_i * grids_weights)
    d2e += contract("Adg,BDg->ABdD", dgamma_dA, f_gamma_A_i * grids_weights)

    return d2e

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
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

    v_ip = numpy.zeros((3,nao,nao))
    vmat = numpy.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'MGGA':
        _check_mgga_grids(grids)
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)
            rks_grad._tau_grad_dot_(v_ip, mol, ao, wv[4], mask, ao_loc, True)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                wv[:,4] *= .25
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)

                for j in range(1, 4):
                    aow = [numint._scale_ao(ao[j], wv[i,4]) for i in range(3)]
                    rks_grad._d1_dot_(vmat[ia], mol, aow, ao[j], mask, ao_loc, True)
            ao_dm0 = aow = None

    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        vmat[ia,:,p0:p1] += v_ip[:,p0:p1]
        vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    return vmat

def _get_vnlc_deriv1_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc Fock matrix 1st derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    nao = mol.nao
    vmat = numpy.empty([mol.natm, 3, nao, nao])

    def get_nlc_vmat(mol, mf, dm):
        ni = mf._numint
        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            assert ni.libxc.is_nlc(mf.nlc)
            xc = mf.nlc
        mf.nlcgrids.build()
        _, _, vnlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, xc, dm)
        return vnlc

    dx = 1e-3
    mol_copy = mol.copy()
    mol_copy.verbose = 0
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_p = get_nlc_vmat(mol_copy, mf, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_m = get_nlc_vmat(mol_copy, mf, dm0)

            vmat[i_atom, i_xyz, :, :] = (vmat_p - vmat_m) / (2 * dx)
    mf.reset(mol)

    return vmat

def get_dweight_dA(mol, grids):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = numpy.asarray(mol.atom_coords(), order = "C")

    from pyscf.dft.gen_grid import original_becke
    assert grids.becke_scheme is original_becke

    radii_adjust = grids.radii_adjust
    atomic_radii = grids.atomic_radii
    if callable(radii_adjust) and atomic_radii is not None:
        f_radii_adjust = radii_adjust(mol, atomic_radii)
        f_radii_table = numpy.asarray([f_radii_adjust(i, j, 0)
                                        for i in range(mol.natm)
                                        for j in range(mol.natm)])
    else:
        f_radii_table = numpy.zeros([mol.natm, mol.natm])

    dweight_dA = numpy.zeros([mol.natm, 3, ngrids], order = "C")
    libdft.VXCbecke_weight_derivative(
        dweight_dA.ctypes.data_as(ctypes.c_void_p),
        grids.coords.ctypes.data_as(ctypes.c_void_p),
        grids.quadrature_weights.ctypes.data_as(ctypes.c_void_p),
        atm_coords.ctypes.data_as(ctypes.c_void_p),
        f_radii_table.ctypes.data_as(ctypes.c_void_p),
        grids.atm_idx.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )
    dweight_dA[grids.atm_idx, 0, numpy.arange(ngrids)] = -numpy.sum(dweight_dA[:, 0, :], axis=0)
    dweight_dA[grids.atm_idx, 1, numpy.arange(ngrids)] = -numpy.sum(dweight_dA[:, 1, :], axis=0)
    dweight_dA[grids.atm_idx, 2, numpy.arange(ngrids)] = -numpy.sum(dweight_dA[:, 2, :], axis=0)

    return dweight_dA

def _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    # Note (Henry Wang 20250428):
    # We observed that in several very simple systems, for example H2O2, H2CO, C2H4,
    # if we do not include the grid response term, the analytical and numerical Fock matrix
    # derivative, although only diff by else than 1e-7 (norm 1), can cause a 1e-3 error in hessian,
    # likely because the CPHF converged to a different solution.
    grid_response = True

    mol = hessobj.mol
    mf = hessobj.base
    natm = mol.natm

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = 2 * mocc @ mocc.T

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = mf._numint.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2)
    # rho_drho = numint.eval_rho(mol, ao[:4, :], dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    ngrids_full = grids.coords.shape[0]
    rho_drho = numpy.empty([4, ngrids_full])

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    ao_nbytes_per_grid = ((4*2) * mol.nao + 4) * 8 # factor of 2 from the ao sorting inside numint.eval_ao()
    ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of CPU memory for NLC Fock first derivative, "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids_full}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids_full, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids_full)
        split_grids_coords = grids.coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1)
        split_rho_drho = numint.eval_rho(mol, split_ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)
        rho_drho[:, g0:g1] = split_rho_drho

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = numpy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = numpy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = numpy.empty(ngrids)
    W_i = numpy.empty(ngrids)
    A_i = numpy.empty(ngrids)
    B_i = numpy.empty(ngrids)
    C_i = numpy.empty(ngrids)
    E_i = numpy.empty(ngrids)

    libdft.VXC_vv10nlc_hessian_eval_UWABCE(
        U_i.ctypes.data_as(ctypes.c_void_p),
        W_i.ctypes.data_as(ctypes.c_void_p),
        A_i.ctypes.data_as(ctypes.c_void_p),
        B_i.ctypes.data_as(ctypes.c_void_p),
        C_i.ctypes.data_as(ctypes.c_void_p),
        E_i.ctypes.data_as(ctypes.c_void_p),
        grids_coords.ctypes.data_as(ctypes.c_void_p),
        grids_weights.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        omega_i.ctypes.data_as(ctypes.c_void_p),
        kappa_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )

    domega_drho_i         = numpy.empty(ngrids)
    domega_dgamma_i       = numpy.empty(ngrids)
    d2omega_drho2_i       = numpy.empty(ngrids)
    d2omega_dgamma2_i     = numpy.empty(ngrids)
    d2omega_drho_dgamma_i = numpy.empty(ngrids)
    libdft.VXC_vv10nlc_hessian_eval_omega_derivative(
        domega_drho_i.ctypes.data_as(ctypes.c_void_p),
        domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        gamma_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i

    aoslices = mol.aoslice_by_atom()
    if grid_response:
        assert grids.atm_idx.shape[0] == grids.coords.shape[0]
        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
        atom_to_grid_index_map = [numpy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2)
    # ao = ao.transpose(0,2,1) # order: component, ao, grid
    # ao_nonzero_rho = ao[:,:,rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)

    # drho_dA, dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices)
    # if grid_response:
    #     drho_dA_grid_response, dgamma_dA_grid_response = \
    #         get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0,
    #                                           atom_to_grid_index_map = atom_to_grid_index_map)
    #     drho_dA   += drho_dA_grid_response
    #     dgamma_dA += dgamma_dA_grid_response
    #     drho_dA_grid_response = None
    #     dgamma_dA_grid_response = None

    drho_dA   = numpy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = numpy.empty([natm, 3, ngrids], order = "C")

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    ao_nbytes_per_grid = ((10 + 1*2 + 3*2 + 9) * mol.nao + (3*2) * mol.natm) * 8
    ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of CPU memory for NLC Fock first derivative, "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        split_grids_coords = grids_coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
        split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

        mu = split_ao[0, :, :]
        dmu_dr = split_ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(split_ao)
        split_drho_dr = nabla_rho_i[:, g0:g1]

        split_drho_dA, split_dgamma_dA = \
            get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices)
        drho_dA  [:, :, g0:g1] = split_drho_dA
        dgamma_dA[:, :, g0:g1] = split_dgamma_dA
        split_drho_dA   = None
        split_dgamma_dA = None

    if grid_response:
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            associated_drho_dr = nabla_rho_i[:, associated_grid_index]

            drho_dA_grid_response   = numpy.empty([3, ngrids_per_atom])
            dgamma_dA_grid_response = numpy.empty([3, ngrids_per_atom])
            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
                split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)
                split_drho_dr = associated_drho_dr[:, g0:g1]
                split_drho_dA_grid_response, split_dgamma_dA_grid_response = \
                    get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, i_atom = i_atom)

                drho_dA_grid_response  [:, g0:g1] =   split_drho_dA_grid_response
                dgamma_dA_grid_response[:, g0:g1] = split_dgamma_dA_grid_response

            drho_dA  [i_atom][:, associated_grid_index] += drho_dA_grid_response
            dgamma_dA[i_atom][:, associated_grid_index] += dgamma_dA_grid_response
            drho_dA_grid_response   = None
            dgamma_dA_grid_response = None

    drho_dA   = numpy.ascontiguousarray(drho_dA)
    dgamma_dA = numpy.ascontiguousarray(dgamma_dA)
    f_rho_A_i   = numpy.empty([natm, 3, ngrids], order = "C")
    f_gamma_A_i = numpy.empty([natm, 3, ngrids], order = "C")

    libdft.VXC_vv10nlc_hessian_eval_f_t(
        f_rho_A_i.ctypes.data_as(ctypes.c_void_p),
        f_gamma_A_i.ctypes.data_as(ctypes.c_void_p),
        grids_coords.ctypes.data_as(ctypes.c_void_p),
        grids_weights.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        omega_i.ctypes.data_as(ctypes.c_void_p),
        kappa_i.ctypes.data_as(ctypes.c_void_p),
        U_i.ctypes.data_as(ctypes.c_void_p),
        W_i.ctypes.data_as(ctypes.c_void_p),
        A_i.ctypes.data_as(ctypes.c_void_p),
        B_i.ctypes.data_as(ctypes.c_void_p),
        C_i.ctypes.data_as(ctypes.c_void_p),
        domega_drho_i.ctypes.data_as(ctypes.c_void_p),
        domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        dkappa_drho_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        d2kappa_drho2_i.ctypes.data_as(ctypes.c_void_p),
        drho_dA.ctypes.data_as(ctypes.c_void_p),
        dgamma_dA.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(3 * natm),
    )
    drho_dA = None
    dgamma_dA = None

    vmat = numpy.zeros([natm, 3, mol.nao, mol.nao])

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2)
    # ao = ao.transpose(0,2,1) # order: component, ao, grid
    # ao_nonzero_rho = ao[:,:,rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)

    # d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    # if grid_response:
    #     d2rho_dAdr_grid_response = get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0,
    #                                                             atom_to_grid_index_map = atom_to_grid_index_map)
    #     d2rho_dAdr += d2rho_dAdr_grid_response
    #     d2rho_dAdr_grid_response = None

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    ao_nbytes_per_grid = ((10 + 1*2 + 3*2 + 9) * mol.nao + (9*2)) * 8
    ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of CPU memory for NLC Fock first derivative, "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for i_atom in range(natm):
        aoslice_one_atom = [aoslices[i_atom]]
        d2rho_dAdr = numpy.empty([3, 3, ngrids])

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
            split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            split_d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslice_one_atom)
            d2rho_dAdr[:, :, g0:g1] = split_d2rho_dAdr
            split_d2rho_dAdr = None

        if grid_response:
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            d2rho_dAdr_grid_response = numpy.empty([3, 3, ngrids_per_atom])
            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
                split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)

                split_d2rho_dAdr_grid_response = \
                    get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, i_atom = i_atom)
                d2rho_dAdr_grid_response[:, :, g0:g1] = split_d2rho_dAdr_grid_response

            d2rho_dAdr[:, :, associated_grid_index] += d2rho_dAdr_grid_response
            split_d2rho_dAdr_grid_response = None

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
            split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            # # w_i 2 f_i^\gamma \nabla_A \nabla\rho \cdot \nabla(\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
            #     'dDg,Dig,jg,g->dij', d2rho_dAdr[i_atom, :, :, :], dmu_dr, mu, f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
            #     'dDg,Dig,jg,g->dji', d2rho_dAdr[i_atom, :, :, :], dmu_dr, mu, f_gamma_i * grids_weights)
            d2rhodAdr_dot_dmudr = contract('dDg,Dig->dig', d2rho_dAdr[:, :, g0:g1], dmu_dr)
            dF  = contract('dig,jg->dij', d2rhodAdr_dot_dmudr, mu * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            d2rhodAdr_dot_dmudr = None

            # # w_i 2 (\nabla\rho)_i \cdot (\nabla(\phi_\mu \phi_nu))_i f_i^{\gamma, A}
            # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
            #     'dg,Dig,jg,Dg->dij', f_gamma_A_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
            # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
            #     'dg,Dig,jg,Dg->dji', f_gamma_A_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
            f_gamma_A_i_mu = contract('dg,ig->dig', f_gamma_A_i[i_atom, :, g0:g1], mu)
            drhodr_dot_dmudr = contract('dig,dg->ig', dmu_dr, split_drho_dr * grids_weights[g0:g1])
            dF += contract('dig,jg->dij', f_gamma_A_i_mu, drhodr_dot_dmudr)
            drhodr_dot_dmudr = None
            f_gamma_A_i_mu = None

            dF += dF.transpose(0,2,1)
            dF *= 2

            # # w_i \phi_{\mu i} \phi_{\nu i} f_i^{\rho, A}
            # vmat[i_atom, :, :, :] += numpy.einsum('dg,ig,jg,g->dij', f_rho_A_i[i_atom, :, :], mu, mu, grids_weights)
            f_rho_A_i_mu = contract('dg,ig->dig', f_rho_A_i[i_atom, :, g0:g1], mu)
            dF += contract('dig,jg->dij', f_rho_A_i_mu, mu * grids_weights[g0:g1])
            f_rho_A_i_mu = None

            vmat[i_atom, :, :, :] += dF
            dF = None

            p0, p1 = aoslices[i_atom][2:]
            # # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, p0:p1, :] += numpy.einsum(
            #     'dig,jg->dij', -dmu_dr[:, p0:p1, :], mu * f_rho_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += numpy.einsum(
            #     'dig,jg->dji', -dmu_dr[:, p0:p1, :], mu * f_rho_i * grids_weights)
            f_rho_dmudA_nu = contract('dig,jg->dij', -dmu_dr[:, p0:p1, :], mu * f_rho_i[g0:g1] * grids_weights[g0:g1])

            # # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, p0:p1, :] += 2 * numpy.einsum(
            #     'dDig,jg,Dg->dij', -d2mu_dr2[:, :, p0:p1, :], mu, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += 2 * numpy.einsum(
            #     'dDig,jg,Dg->dji', -d2mu_dr2[:, :, p0:p1, :], mu, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, p0:p1, :] += 2 * numpy.einsum(
            #     'dig,Djg,Dg->dij', -dmu_dr[:, p0:p1, :], dmu_dr, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += 2 * numpy.einsum(
            #     'dig,Djg,Dg->dji', -dmu_dr[:, p0:p1, :], dmu_dr, nabla_rho_i * f_gamma_i * grids_weights)
            mu_dot_drhodr = contract('ig,dg->dig', mu, split_drho_dr * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            f_gamma_d2mudr2_nu = contract('dDig,Djg->dij', -d2mu_dr2[:, :, p0:p1, :], mu_dot_drhodr)
            mu_dot_drhodr = None
            dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            f_gamma_dmudr_dnudr = contract('dig,jg->dij', -dmu_dr[:, p0:p1, :], dmudr_dot_drhodr)
            dmudr_dot_drhodr = None

            dF_ao = f_rho_dmudA_nu + 2 * (f_gamma_d2mudr2_nu + f_gamma_dmudr_dnudr)
            f_rho_dmudA_nu = None
            f_gamma_d2mudr2_nu = None
            f_gamma_dmudr_dnudr = None

            vmat[i_atom, :, p0:p1, :] += dF_ao
            vmat[i_atom, :, :, p0:p1] += dF_ao.transpose(0,2,1)
            dF_ao = None

        d2rho_dAdr = None

        if grid_response:
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            associated_drho_dr = nabla_rho_i[:, associated_grid_index]
            fw_rho_associated_grids   =   f_rho_i[associated_grid_index] * grids_weights[associated_grid_index]
            fw_gamma_associated_grids = f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index]

            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
                split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)
                split_drho_dr = associated_drho_dr[:, g0:g1]

                # # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
                # vmat[i_atom, :, :, :] += numpy.einsum('dig,jg->dij',
                #     dmu_dr[:, :, associated_grid_index],
                #     mu[:, associated_grid_index] * fw_rho_associated_grids)
                # vmat[i_atom, :, :, :] += numpy.einsum('dig,jg->dji',
                #     dmu_dr[:, :, associated_grid_index],
                #     mu[:, associated_grid_index] * fw_rho_associated_grids)
                f_rho_dmudA_nu = contract('dig,jg->dij', dmu_dr, mu * fw_rho_associated_grids[g0:g1])

                # # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dDig,jg,Dg->dij',
                #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * fw_gamma_associated_grids)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dDig,jg,Dg->dji',
                #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * fw_gamma_associated_grids)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dig,Djg,Dg->dij',
                #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * fw_gamma_associated_grids)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dig,Djg,Dg->dji',
                #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * fw_gamma_associated_grids)
                d2mudr2_dot_drhodr = contract('dDig,Dg->dig',
                                              d2mu_dr2, split_drho_dr * fw_gamma_associated_grids[g0:g1])
                f_gamma_d2mudr2_nu = contract('dig,jg->dij', d2mudr2_dot_drhodr, mu)
                d2mudr2_dot_drhodr = None
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * fw_gamma_associated_grids[g0:g1])
                f_gamma_dmudr_dnudr = contract('dig,jg->dij', dmu_dr, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None

                dF_ao = f_rho_dmudA_nu + 2 * (f_gamma_d2mudr2_nu + f_gamma_dmudr_dnudr)
                f_rho_dmudA_nu = None
                f_gamma_d2mudr2_nu = None
                f_gamma_dmudr_dnudr = None

                dF_ao += dF_ao.transpose(0,2,1)

                vmat[i_atom, :, :, :] += dF_ao
                dF_ao = None

    if grid_response:
        E_Bgr_i = numpy.empty([natm, 3, ngrids], order = "C")
        U_Bgr_i = numpy.empty([natm, 3, ngrids], order = "C")
        W_Bgr_i = numpy.empty([natm, 3, ngrids], order = "C")
        libdft.VXC_vv10nlc_hessian_eval_EUW_grid_response(
            E_Bgr_i.ctypes.data_as(ctypes.c_void_p),
            U_Bgr_i.ctypes.data_as(ctypes.c_void_p),
            W_Bgr_i.ctypes.data_as(ctypes.c_void_p),
            grids_coords.ctypes.data_as(ctypes.c_void_p),
            grids_weights.ctypes.data_as(ctypes.c_void_p),
            rho_i.ctypes.data_as(ctypes.c_void_p),
            omega_i.ctypes.data_as(ctypes.c_void_p),
            kappa_i.ctypes.data_as(ctypes.c_void_p),
            grid_to_atom_index_map.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )

        grids_weights_1 = get_dweight_dA(mol, grids)
        grids_weights_1 = grids_weights_1[:, :, rho_nonzero_mask]
        grids_weights_1 = numpy.ascontiguousarray(grids_weights_1)

        E_Bw_i = numpy.empty([natm, 3, ngrids], order = "C")
        U_Bw_i = numpy.empty([natm, 3, ngrids], order = "C")
        W_Bw_i = numpy.empty([natm, 3, ngrids], order = "C")
        libdft.VXC_vv10nlc_hessian_eval_EUW_with_weight1(
            E_Bw_i.ctypes.data_as(ctypes.c_void_p),
            U_Bw_i.ctypes.data_as(ctypes.c_void_p),
            W_Bw_i.ctypes.data_as(ctypes.c_void_p),
            grids_coords.ctypes.data_as(ctypes.c_void_p),
            grids_weights_1.ctypes.data_as(ctypes.c_void_p),
            rho_i.ctypes.data_as(ctypes.c_void_p),
            omega_i.ctypes.data_as(ctypes.c_void_p),
            kappa_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm * 3),
        )

        f_rho_grid_response_i = (E_Bw_i + E_Bgr_i) \
            + ((U_Bw_i + U_Bgr_i) * dkappa_drho_i + (W_Bw_i + W_Bgr_i) * domega_drho_i) * rho_i
        f_gamma_grid_response_i = (W_Bw_i + W_Bgr_i) * domega_dgamma_i * rho_i
        E_Bw_i = None
        U_Bw_i = None
        W_Bw_i = None
        E_Bgr_i = None
        U_Bgr_i = None
        W_Bgr_i = None

        mem_now = lib.current_memory()[0]
        available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
        ao_nbytes_per_grid = ((4 + 1*2 + 3*2) * mol.nao) * 8
        ngrids_per_batch = int(available_cpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of CPU memory for NLC Fock first derivative, "
                              f"available cpu memory = {available_cpu_memory}"
                              f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2)
            split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            for i_atom in range(natm):
                # # \nabla_A w_i term
                # vmat[i_atom, :, :, :] += numpy.einsum(
                #     'dg,ig,jg->dij', grids_weights_1[i_atom, :, :], mu, mu * f_rho_i)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
                #     'dg,Dig,jg,Dg->dij', grids_weights_1[i_atom, :, :], dmu_dr, mu, nabla_rho_i * f_gamma_i)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum(
                #     'dg,Dig,jg,Dg->dji', grids_weights_1[i_atom, :, :], dmu_dr, mu, nabla_rho_i * f_gamma_i)
                dwdr_dot_mu = contract('dg,ig->dig', grids_weights_1[i_atom, :, g0:g1], mu)
                f_rho_dwdr  = contract('dig,jg->dij', dwdr_dot_mu, mu * f_rho_i[g0:g1])
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * f_gamma_i[g0:g1])
                f_gamma_dwdr  = contract('dig,jg->dij', dwdr_dot_mu, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None
                dwdr_dot_mu = None

                # # E_i^{Aw} and E_i^{Agr} terms combined
                # vmat[i_atom, :, :, :] += numpy.einsum(
                #     'dg,ig,jg->dij', f_rho_grid_response_i[i_atom, :, :], mu, mu * grids_weights)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dg,Dig,jg,Dg->dij',
                #     f_gamma_grid_response_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
                # vmat[i_atom, :, :, :] += 2 * numpy.einsum('dg,Dig,jg,Dg->dji',
                #     f_gamma_grid_response_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
                dfrhodr_dot_mu = contract('dg,ig->dig', f_rho_grid_response_i[i_atom, :, g0:g1], mu)
                f_rho_dwdr += contract('dig,jg->dij', dfrhodr_dot_mu, mu * grids_weights[g0:g1])
                dfrhodr_dot_mu = None
                dfgammadr_dot_mu = contract('dg,ig->dig', f_gamma_grid_response_i[i_atom, :, g0:g1], mu)
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * grids_weights[g0:g1])
                f_gamma_dwdr += contract('dig,jg->dij', dfgammadr_dot_mu, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None
                dfgammadr_dot_mu = None

                f_gamma_dwdr += f_gamma_dwdr.transpose(0,2,1)
                dF_ao = f_rho_dwdr + 2 * f_gamma_dwdr
                f_rho_dwdr = None
                f_gamma_dwdr = None

                vmat[i_atom, :, :, :] += dF_ao
                dF_ao = None

    return vmat

def get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1s, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.

        mo_coeff, mo_occ are 0-th order
        dm1s is first order

        TODO: check the effect of different grid, using mf.nlcgrids right now
    """
    if mo_coeff.ndim == 2:
        mocc = mo_coeff[:,mo_occ>0]
        mo_occ = mo_occ[mo_occ > 0]
        dm0 = (mocc * mo_occ) @ mocc.T
    else:
        assert mo_coeff.ndim == 3 # unrestricted case
        assert mo_coeff.shape[0] == 2
        assert mo_occ.shape[0] == 2
        mocc_a = mo_coeff[0][:, mo_occ[0] > 0]
        mocc_b = mo_coeff[1][:, mo_occ[1] > 0]
        mo_occ_a = mo_occ[0, mo_occ[0] > 0]
        mo_occ_b = mo_occ[1, mo_occ[1] > 0]
        dm0 = (mocc_a * mo_occ_a) @ mocc_a.T + (mocc_b * mo_occ_b) @ mocc_b.T

    output_in_2d = False
    if dm1s.ndim == 2:
        assert dm1s.shape == (mol.nao, mol.nao)
        dm1s = dm1s.reshape((1, mol.nao, mol.nao))
        output_in_2d = True
    assert dm1s.ndim == 3

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    n_dm1 = dm1s.shape[0]

    ni = mf._numint

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]

    # ao = numint.eval_ao(mol, grids.coords, deriv = 1)
    # rho_drho = numint.eval_rho(mol, ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    ngrids_full = grids.coords.shape[0]
    rho_drho = numpy.empty([4, ngrids_full])
    g1 = 0
    for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(mol, grids, mol.nao, 1, max_memory):
        g0, g1 = g1, g1 + split_weights.size
        rho_drho[:, g0:g1] = numint.eval_rho(mol, split_ao, dm0, xctype = "NLC", hermi = 1)
    dm0 = None

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = numpy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = numpy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = numpy.empty(ngrids)
    W_i = numpy.empty(ngrids)
    A_i = numpy.empty(ngrids)
    B_i = numpy.empty(ngrids)
    C_i = numpy.empty(ngrids)
    E_i = numpy.empty(ngrids) # Not used

    libdft.VXC_vv10nlc_hessian_eval_UWABCE(
        U_i.ctypes.data_as(ctypes.c_void_p),
        W_i.ctypes.data_as(ctypes.c_void_p),
        A_i.ctypes.data_as(ctypes.c_void_p),
        B_i.ctypes.data_as(ctypes.c_void_p),
        C_i.ctypes.data_as(ctypes.c_void_p),
        E_i.ctypes.data_as(ctypes.c_void_p),
        grids_coords.ctypes.data_as(ctypes.c_void_p),
        grids_weights.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        omega_i.ctypes.data_as(ctypes.c_void_p),
        kappa_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )
    E_i = None

    domega_drho_i         = numpy.empty(ngrids)
    domega_dgamma_i       = numpy.empty(ngrids)
    d2omega_drho2_i       = numpy.empty(ngrids)
    d2omega_dgamma2_i     = numpy.empty(ngrids)
    d2omega_drho_dgamma_i = numpy.empty(ngrids)
    libdft.VXC_vv10nlc_hessian_eval_omega_derivative(
        domega_drho_i.ctypes.data_as(ctypes.c_void_p),
        domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
        d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
        rho_i.ctypes.data_as(ctypes.c_void_p),
        gamma_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_gamma_i = rho_i * domega_dgamma_i * W_i

    # ao = numint.eval_ao(mol, grids.coords, deriv = 1)
    # rho_drho_t = numpy.empty([n_dm1, 4, ngrids])
    # for i_dm in range(n_dm1):
    #     dm1 = dm1s[i_dm, :, :]
    #     rho_drho_1 = numint.eval_rho(mol, ao, dm1, xctype = "NLC", hermi = 0, with_lapl = False)
    #     rho_drho_t[i_dm, :, :] = rho_drho_1[:, rho_nonzero_mask]

    vmat = numpy.zeros([n_dm1, mol.nao, mol.nao])

    mem_now = lib.current_memory()[0]
    available_cpu_memory = max(16e3, max_memory * 0.5 - mem_now) * 1e6
    fxc_nbytes_per_dm1 = ((1*6 + 3*2) * ngrids + (1*2 + 3*2) * ngrids_full) * 8
    ndm1_per_batch = int(available_cpu_memory / fxc_nbytes_per_dm1)
    if ndm1_per_batch < 6:
        raise MemoryError(f"Out of CPU memory for NLC response (orbital hessian), "
                          f"available cpu memory = {available_cpu_memory}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ndm1_per_batch = (ndm1_per_batch + 6 - 1) // 6 * 6

    for i_dm1_batch in range(0, n_dm1, ndm1_per_batch):
        n_dm1_batch = min(ndm1_per_batch, n_dm1 - i_dm1_batch)

        rho_drho_t = numpy.empty([n_dm1_batch, 4, ngrids_full])
        g1 = 0
        for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(mol, grids, mol.nao, 1, max_memory):
            g0, g1 = g1, g1 + split_weights.size
            for i_dm in range(n_dm1_batch):
                dm1_subset = dm1s[i_dm + i_dm1_batch, :, :]
                rho_drho_t[i_dm, :, g0:g1] = numint.eval_rho(mol, split_ao, dm1_subset, xctype = "NLC", hermi = 0)
                dm1_subset = None
        rho_drho_t = rho_drho_t[:, :, rho_nonzero_mask]

        rho_t_i = rho_drho_t[:, 0, :]
        nabla_rho_t_i = rho_drho_t[:, 1:4, :]
        gamma_t_i = nabla_rho_i[0, :] * nabla_rho_t_i[:, 0, :] \
                    + nabla_rho_i[1, :] * nabla_rho_t_i[:, 1, :] \
                    + nabla_rho_i[2, :] * nabla_rho_t_i[:, 2, :]
        gamma_t_i *= 2 # Account for the factor of 2 before gamma_j^t term in equation (22)
        rho_drho_t = None

        rho_t_i   = numpy.ascontiguousarray(rho_t_i)
        gamma_t_i = numpy.ascontiguousarray(gamma_t_i)
        f_rho_t_i   = numpy.empty([n_dm1_batch, ngrids], order = "C")
        f_gamma_t_i = numpy.empty([n_dm1_batch, ngrids], order = "C")

        libdft.VXC_vv10nlc_hessian_eval_f_t(
            f_rho_t_i.ctypes.data_as(ctypes.c_void_p),
            f_gamma_t_i.ctypes.data_as(ctypes.c_void_p),
            grids_coords.ctypes.data_as(ctypes.c_void_p),
            grids_weights.ctypes.data_as(ctypes.c_void_p),
            rho_i.ctypes.data_as(ctypes.c_void_p),
            omega_i.ctypes.data_as(ctypes.c_void_p),
            kappa_i.ctypes.data_as(ctypes.c_void_p),
            U_i.ctypes.data_as(ctypes.c_void_p),
            W_i.ctypes.data_as(ctypes.c_void_p),
            A_i.ctypes.data_as(ctypes.c_void_p),
            B_i.ctypes.data_as(ctypes.c_void_p),
            C_i.ctypes.data_as(ctypes.c_void_p),
            domega_drho_i.ctypes.data_as(ctypes.c_void_p),
            domega_dgamma_i.ctypes.data_as(ctypes.c_void_p),
            dkappa_drho_i.ctypes.data_as(ctypes.c_void_p),
            d2omega_drho2_i.ctypes.data_as(ctypes.c_void_p),
            d2omega_dgamma2_i.ctypes.data_as(ctypes.c_void_p),
            d2omega_drho_dgamma_i.ctypes.data_as(ctypes.c_void_p),
            d2kappa_drho2_i.ctypes.data_as(ctypes.c_void_p),
            rho_t_i.ctypes.data_as(ctypes.c_void_p),
            gamma_t_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(n_dm1_batch),
        )
        rho_t_i = None
        gamma_t_i = None

        fxc_rho = f_rho_t_i * grids_weights
        f_rho_t_i = None
        fxc_gamma  = contract("dg,tg->tdg", nabla_rho_i, f_gamma_t_i)
        f_gamma_t_i = None
        fxc_gamma += nabla_rho_t_i * f_gamma_i
        nabla_rho_t_i = None
        fxc_gamma = 2 * fxc_gamma * grids_weights

        fxc_rho_full = numpy.zeros([n_dm1_batch, ngrids_full])
        fxc_rho_full[:, rho_nonzero_mask] = fxc_rho
        fxc_rho = None
        fxc_gamma_full = numpy.zeros([n_dm1_batch, 3, ngrids_full])
        fxc_gamma_full[:, :, rho_nonzero_mask] = fxc_gamma
        fxc_gamma = None

        g1 = 0
        for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(mol, grids, mol.nao, 1, max_memory):
            split_ao = split_ao.transpose(0,2,1) # order: component, ao, grid
            g0, g1 = g1, g1 + split_weights.size
            split_fxc_rho = fxc_rho_full[:, g0:g1]
            split_fxc_gamma = fxc_gamma_full[:, :, g0:g1]

            for i_dm in range(n_dm1_batch):
                # \mu \nu
                V_munu = contract("ig,jg->ij", split_ao[0], split_ao[0] * split_fxc_rho[i_dm, :])

                # \mu \nabla\nu + \nabla\mu \nu
                nabla_fxc_dot_nabla_ao = contract("dg,dig->ig", split_fxc_gamma[i_dm, :, :], split_ao[1:4])
                V_munu_gamma = contract("ig,jg->ij", split_ao[0], nabla_fxc_dot_nabla_ao)
                nabla_fxc_dot_nabla_ao = None
                V_munu += V_munu_gamma
                V_munu += V_munu_gamma.T
                V_munu_gamma = None

                vmat[i_dm + i_dm1_batch, :, :] += V_munu
                V_munu = None

    if output_in_2d:
        vmat = vmat.reshape((mol.nao, mol.nao))

    return vmat

def _check_mgga_grids(grids):
    mol = grids.mol
    atom_grid = grids.atom_grid
    if atom_grid:
        if isinstance(atom_grid, (tuple, list)):
            n_rad = atom_grid[0]
            if n_rad < 150 and any(mol.atom_charges() > 10):
                logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                            f'{atom_grid} may not be dense enough.')
        else:
            symbols = [mol.atom_symbol(ia) for ia in range(mol.natm)]
            problematic = []
            for symb in symbols:
                chg = gto.charge(symb)
                if symb in atom_grid:
                    n_rad = atom_grid[symb][0]
                else:
                    n_rad = gen_grid._default_rad(chg, grids.level)
                if n_rad < 150 and chg > 10:
                    problematic.append((symb, n_rad))
            if problematic:
                problematic = [f'{symb}: {r}' for symb, r in problematic]
                logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                            f'Radial grids {",".join(problematic)} '
                            'may not be dense enough.')
    elif grids.level < 5:
        logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                    f'grids.level {grids.level} may not be dense enough.')


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic RKS hessian'''

    _keys = {'grids', 'grid_response'}

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    partial_hess_elec = partial_hess_elec
    hess_elec = rhf_hess.hess_elec
    make_h1 = make_h1

from pyscf import dft
dft.rks.RKS.Hessian = dft.rks_symm.RKS.Hessian = lib.class_as_method(Hessian)
dft.roks.ROKS.Hessian = dft.rks_symm.ROKS.Hessian = lib.invalid_method('Hessian')
