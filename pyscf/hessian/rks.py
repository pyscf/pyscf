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

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint


# import pyscf.grad.rks to activate nuc_grad_method method
from pyscf.grad import rks  # noqa


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, beta = mf._numint.rsh_coeff(mf.xc)
    if abs(omega) > 1e-10:
        hyb = alpha + beta
    else:
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=mol.spin)
    de2, ej, ek = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             abs(hyb) > 1e-10)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if abs(omega) > 1e-10:
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
        if abs(omega) > 1e-10:
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

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb) > 1e-10:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if abs(omega) > 1e-10:
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

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile

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
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
            aow = None

    elif xctype == 'GGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numpy.einsum('pi,p->pi', ao[aoidx[0]], wv[1])
            aow+= numpy.einsum('pi,p->pi', ao[aoidx[1]], wv[2])
            aow+= numpy.einsum('pi,p->pi', ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            # *2 because v.T is not applied. Only v is computed in the next _dot_ao_ao 
            wv[0] *= 2
            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
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
        raise NotImplementedError('meta-GGA')

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[0]
    rho1 = numpy.zeros((3,4,ngrids))
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
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            aow = numpy.einsum('xpi,p->xpi', ao[1:4], weight*vrho)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                # aow ~ rho1 ~ d/dR1
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = numint._rks_gga_wv1(rho, dR_rho1[0], vxc, fxc, weight)
                wv[1] = numint._rks_gga_wv1(rho, dR_rho1[1], vxc, fxc, weight)
                wv[2] = numint._rks_gga_wv1(rho, dR_rho1[2], vxc, fxc, weight)

                aow = rks_grad._make_dR_dao_w(ao, wv[0])
                rks_grad._d1_dot_(vmat[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wv[1])
                rks_grad._d1_dot_(vmat[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = rks_grad._make_dR_dao_w(ao, wv[2])
                rks_grad._d1_dot_(vmat[ia,2], mol, aow, ao[0], mask, ao_loc, True)

                aow = numpy.einsum('npi,Xnp->Xpi', ao[:4], wv)
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmat

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

    vmat = numpy.zeros((mol.natm,3,nao,nao))
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
            aow1 = numpy.einsum('xpi,p->xpi', ao[1:], weight*vrho)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                aow[:,:,p0:p1] += aow1[:,:,p0:p1]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = aow1 = None

        for ia in range(mol.natm):
            vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        v_ip = numpy.zeros((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = numint._rks_gga_wv1(rho, dR_rho1[0], vxc, fxc, weight)
                wv[1] = numint._rks_gga_wv1(rho, dR_rho1[1], vxc, fxc, weight)
                wv[2] = numint._rks_gga_wv1(rho, dR_rho1[2], vxc, fxc, weight)
                aow = numpy.einsum('npi,Xnp->Xpi', ao[:4], wv)
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,p0:p1] += v_ip[:,p0:p1]
            vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmat


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grids'])

    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1

from pyscf import dft
dft.rks.RKS.Hessian = dft.rks_symm.RKS.Hessian = lib.class_as_method(Hessian)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    #dft.numint.NumInt.libxc = dft.xcfun
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
        [1 , (0. ,  1.517  , 1.177)],
        ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = dft.RKS(mol)
    mf.grids.level = 4
    mf.grids.prune = False
    mf.xc = xc_code
    mf.conv_tol = 1e-14
    mf.kernel()
    n3 = mol.natm * 3
    hobj = mf.Hessian()
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.finger(e2) - -0.42286447944621297)
    print(lib.finger(e2) - -0.45453541215680582)
    print(lib.finger(e2) - -0.41385249055285972)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            mf = dft.RKS(mol).set(conv_tol=1e-14, xc=xc_code).run()
            e1a = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = dft.RKS(mol).set(conv_tol=1e-14, xc=xc_code).run()
            e1b = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-4) for ia in range(mol.natm)]
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
    print(numpy.allclose(e2,e2ref,atol=1e-8))
