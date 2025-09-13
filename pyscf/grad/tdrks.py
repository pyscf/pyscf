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
# Ref:
# J. Chem. Phys. 117, 7433
#


from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import rks
from pyscf.dft import numint
from pyscf.scf import cphf
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf


#
# Given Y = 0, TDDFT gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    assert td_grad.base.frozen is None

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, dmxpy,
                                dmzoo, True, True, singlet, max_memory)

    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if omega != 0:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = f1vo[0] - vk[1]
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2
    else:
        vj = mf.get_j(mol, (dmzoo, dmxpy+dmxpy.T), hermi=1)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0]
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff0mom = numpy.zeros((nmo,nmo))

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1,
                            with_nlc=not td_grad.base.exclude_nlc)
    def fvind(x):
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(numpy.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao+z1ao.T)

    im0 = numpy.zeros((nmo,nmo))
    im0[:nocc,:nocc] = reduce(numpy.dot, (orbo.T, veff0doo+veff, orbo))
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy)
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,nocc:] = numpy.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy)
    im0[nocc:,nocc:]+= numpy.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,:nocc] = numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy)*2
    im0[nocc:,:nocc]+= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy)*2

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo
    dm1[nocc:,nocc:] = dvv
    dm1[nocc:,:nocc] = z1
    dm1[:nocc,:nocc] += numpy.eye(nocc)*2 # for ground state
    im0 = reduce(numpy.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo
    oo0 = reduce(numpy.dot, (orbo, orbo.T))
    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += td_grad.get_k(mol, dm, omega=omega) * (alpha-hyb)
        vj = vj.reshape(-1,3,nao,nao)
        vk = vk.reshape(-1,3,nao,nao)
        veff1 = -vk
        if singlet:
            veff1 += vj * 2
        else:
            veff1[:2] += vj[:2] * 2
    else:
        vj = td_grad.get_j(mol, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T))
        vj = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        if singlet:
            veff1[:3] = vj * 2
        else:
            veff1[:2] = vj[:2] * 2

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                False, False, True, max_memory)[0]

    veff1[0] += vxc1[1:]
    veff1[1] +=(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    if singlet:
        veff1[2] += f1vo[1:] * 2
    else:
        veff1[2] += f1vo[1:]
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += veff1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += veff1[0,:,p0:p1].transpose(0,2,1)
        # oo0*2 for doubly occupied orbitals
        e1  = numpy.einsum('xpq,pq->x', h1ao, oo0) * 2

        e1 += numpy.einsum('xpq,pq->x', h1ao, dmz1doo)
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        e1 -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        e1 += numpy.einsum('xij,ij->x', veff1[1,:,p0:p1], oo0[p0:p1])
        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], dmxpy[p0:p1,:]) * 2
        e1 += numpy.einsum('xji,ij->x', veff1[2,:,p0:p1], dmxpy[:,p0:p1]) * 2
        e1 += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], dmxmy[p0:p1,:]) * 2
        e1 -= numpy.einsum('xji,ij->x', veff1[3,:,p0:p1], dmxmy[:,p0:p1]) * 2

        e1 += td_grad.extra_force(ia, locals())

        de[k] = e1

    log.timer('TDRKS nuclear gradients', *time0)
    return de


# dmvo, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
#TODO: to include the response of grids
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, singlet=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * .5 # because K_{ia,jb} == K_{ia,bj}

    f1vo = numpy.zeros((4,nao,nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = numpy.zeros((4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = numpy.zeros((4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = numpy.zeros((4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDRKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    if mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

    if singlet:
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1,
                               with_lapl=False) * 2  # *2 for alpha + beta
            if xctype == 'LDA':
                rho1 = rho1[numpy.newaxis]
            wv = numpy.einsum('yg,xyg,g->xg', rho1, fxc, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False) * 2
                if xctype == 'LDA':
                    rho2 = rho2[numpy.newaxis]
                wv = numpy.einsum('yg,xyg,g->xg', rho2, fxc, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                wv = numpy.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype, with_lapl=False)
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            # fxc_t couples triplet excitation amplitudes
            # 1/2 int (tia - tIA) fxc (tjb - tJB) = tia fxc_t tjb
            fxc_t = fxc[:,:,0] - fxc[:,:,1]
            fxc_t = fxc_t[0] - fxc_t[1]

            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[numpy.newaxis]
            wv = numpy.einsum('yg,xyg,g->xg', rho1, fxc_t, weight)
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                # fxc_s == 2 * fxc of spin restricted xc kernel
                # provides f1oo to couple the interaction between first order MO
                # and density response of tddft amplitudes, which is described by dmoo
                fxc_s = fxc[0,:,0] + fxc[0,:,1]
                rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1, with_lapl=False)
                if xctype == 'LDA':
                    rho2 = rho2[numpy.newaxis]
                wv = numpy.einsum('yg,xyg,g->xg', rho2, fxc_s, weight)
                fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                vxc = vxc[0]
                fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                # kxc in terms of the triplet coupling
                # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
                kxc = kxc[0,:,0] - kxc[0,:,1]
                kxc = kxc[:,:,0] - kxc[:,:,1]
                wv = numpy.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None: f1oo[1:] *= -1
    if v1ao is not None: v1ao[1:] *= -1
    if k1ao is not None: k1ao[1:] *= -1
    return f1vo, f1oo, v1ao, k1ao

def _lda_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    aow = numint._scale_ao(ao[0], wv[0])
    for k in range(4):
        vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
    return vmat

def _gga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= .5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vmat[0] += tmp + tmp.T
    rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)
    return vmat

def _mgga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= .5  # *.5 because vmat + vmat.T at the end
    wv[4] *= .5  # *.5 for 1/2 in tau
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vmat[0] += tmp + tmp.T
    vmat[0] += numint._tau_dot(mol, ao, ao, wv[4], mask, shls_slice, ao_loc)
    rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv[:4], mask, ao_loc)
    rks_grad._tau_grad_dot_(vmat[1:], mol, ao, wv[4], mask, ao_loc, True)
    return vmat


class Gradients(tdrhf.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

Grad = Gradients
