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

import time
import copy
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import dft
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
    log = logger.new_logger(td_grad, verbose)
    time0 = time.clock(), time.time()

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
    dmzvop = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmzvom = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    dm0 = None
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, dmzvop,
                                dmzoo, True, True, singlet, max_memory)

    if abs(hyb) > 1e-10:
        dm = (dmzoo, dmzvop+dmzvop.T, dmzvom-dmzvom.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if abs(omega) > 1e-10:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = -vk[1] + f1vo[0] * 2
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2
    else:
        vj = mf.get_j(mol, (dmzoo, dmzvop+dmzvop.T), hermi=1)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0] * 2
        veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff0mom = numpy.zeros((nmo,nmo))
    def fvind(x):
# Cannot make call to .base.get_vind because first order orbitals are solved
# through closed shell ground state CPHF.
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc), orbo.T))
        dm = dm + dm.T
# Call singlet XC kernel contraction, for closed shell ground state
        vindxc = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm, 0,
                                      singlet, rho0, vxc, fxc, max_memory)
        if abs(hyb) > 1e-10:
            vj, vk = mf.get_jk(mol, dm)
            veff = vj * 2 - hyb * vk + vindxc
            if abs(omega) > 1e-10:
                veff -= mf.get_k(mol, dm, hermi=1, omega=omega) * (alpha-hyb)
        else:
            vj = mf.get_j(mol, dm)
            veff = vj * 2 + vindxc
        return reduce(numpy.dot, (orbv.T, veff, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao  = reduce(numpy.dot, (orbv, z1, orbo.T))
# Note Z-vector is always associated to singlet integrals.
    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                False, False, True, max_memory)[0]
    if abs(hyb) > 1e-10:
        vj, vk = mf.get_jk(mol, z1ao, hermi=0)
        veff = vj * 2 - hyb * vk + fxcz1[0]
        if abs(omega) > 1e-10:
            veff -= mf.get_k(mol, z1ao, hermi=0, omega=omega) * (alpha-hyb)
    else:
        vj = mf.get_j(mol, z1ao, hermi=1)
        veff = vj * 2 + fxcz1[0]

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
    if abs(hyb) > 1e-10:
        dm = (oo0, dmz1doo+dmz1doo.T, dmzvop+dmzvop.T, dmzvom-dmzvom.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vk *= hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dm) * (alpha-hyb)
        vj = vj.reshape(-1,3,nao,nao)
        vk = vk.reshape(-1,3,nao,nao)
        if singlet:
            veff1 = vj * 2 - vk
        else:
            veff1 = numpy.vstack((vj[:2]*2-vk[:2], -vk[2:]))
    else:
        vj = td_grad.get_j(mol, (oo0, dmz1doo+dmz1doo.T, dmzvop+dmzvop.T))
        vj = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        if singlet:
            veff1[:3] = vj * 2
        else:
            veff1[:2] = vj[:2] * 2
    veff1[0] += vxc1[1:]
    veff1[1] +=(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    veff1[2] += f1vo[1:] * 2
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
        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], dmzvop[p0:p1,:]) * 2
        e1 += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], dmzvom[p0:p1,:]) * 2
        e1 += numpy.einsum('xji,ij->x', veff1[2,:,p0:p1], dmzvop[:,p0:p1]) * 2
        e1 -= numpy.einsum('xji,ij->x', veff1[3,:,p0:p1], dmzvom[:,p0:p1]) * 2

        de[k] = e1

    log.timer('TDDFT nuclear gradients', *time0)
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
    dmvo = (dmvo + dmvo.T) * .5 # because K_{ia,jb} == K_{ia,jb}

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

    if xctype == 'LDA':
        ao_deriv = 1
        if singlet:
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
                vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 0, deriv=deriv)[1:]

                wfxc = fxc[0] * weight * 2  # *2 for alpha+beta
                rho1 = ni.eval_rho(mol, ao[0], dmvo, mask, 'LDA')
                aow = numpy.einsum('pi,p,p->pi', ao[0], wfxc, rho1)
                for k in range(4):
                    f1vo[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
                if dmoo is not None:
                    rho2 = ni.eval_rho(mol, ao[0], dmoo, mask, 'LDA')
                    aow = numpy.einsum('pi,p,p->pi', ao[0], wfxc, rho2)
                    for k in range(4):
                        f1oo[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
                if with_vxc:
                    aow = numpy.einsum('pi,p,p->pi', ao[0], vxc[0], weight)
                    for k in range(4):
                        v1ao[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
                if with_kxc:
                    aow = numpy.einsum('pi,p,p,p->pi', ao[0], kxc[0], weight, rho1**2)
                    for k in range(4):
                        k1ao[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
                vxc = fxc = kxc = aow = rho = rho1 = rho2 = None
            if with_kxc:  # for (rho1*2)^2, *2 for alpha+beta in singlet
                k1ao *= 4

        else:
            raise NotImplementedError('LDA triplet')

    elif xctype == 'GGA':
        if singlet:
            def gga_sum_(vmat, ao, wv, mask):
                aow  = numpy.einsum('pi,p->pi', ao[0], wv[0])
                aow += numpy.einsum('npi,np->pi', ao[1:4], wv[1:])
                tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                vmat[0] += tmp + tmp.T
                rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)
            ao_deriv = 2
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'GGA')
                vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 0, deriv=deriv)[1:]

                rho1 = ni.eval_rho(mol, ao, dmvo, mask, 'GGA') * 2  # *2 for alpha + beta
                wv = numint._rks_gga_wv1(rho, rho1, vxc, fxc, weight)
                gga_sum_(f1vo, ao, wv, mask)

                if dmoo is not None:
                    rho2 = ni.eval_rho(mol, ao, dmoo, mask, 'GGA') * 2
                    wv = numint._rks_gga_wv1(rho, rho2, vxc, fxc, weight)
                    gga_sum_(f1oo, ao, wv, mask)
                if with_vxc:
                    wv = numint._rks_gga_wv0(rho, vxc, weight)
                    gga_sum_(v1ao, ao, wv, mask)
                if with_kxc:
                    wv = numint._rks_gga_wv2(rho, rho1, fxc, kxc, weight)
                    gga_sum_(k1ao, ao, wv, mask)
                vxc = fxc = kxc = rho = rho1 = None

        else:
            raise NotImplementedError('GGA triplet')

    else:
        raise NotImplementedError('meta-GGA')

    f1vo[1:] *= -1
    if f1oo is not None: f1oo[1:] *= -1
    if v1ao is not None: v1ao[1:] *= -1
    if k1ao is not None: k1ao[1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


class Gradients(tdrhf.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

Grad = Gradients

from pyscf import tdscf
tdscf.rks.TDA.Gradients = tdscf.rks.TDDFT.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import dft
    from pyscf import tddft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'LDA,'
    mf.grids.prune = False
    mf.conv_tol = 1e-14
#    mf.grids.level = 6
    mf.scf()

    td = tddft.TDDFT(mf)
    td.nstates = 3
    e, z = td.kernel()
    tdg = td.Gradients()
    g1 = tdg.kernel(state=3)
    print(g1)
# [[ 0  0  -1.31315477e-01]
#  [ 0  0   1.31319442e-01]]
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[2]-e2[2])/.002 - g1[0,2]).max())

    mol.set_geom_('H 0 0 1.804; F 0 0 0', unit='B')
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf._numint.libxc = dft.xcfun
    mf.grids.prune = False
    mf.conv_tol = 1e-14
    mf.scf()

    td = tddft.TDA(mf)
    td.nstates = 3
    e, z = td.kernel()
    tdg = td.Gradients()
    g1 = tdg.kernel(state=3)
    print(g1)
# [[ 0  0  -1.21504524e-01]
#  [ 0  0   1.21505341e-01]]
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[2]-e2[2])/.002 - g1[0,2]).max())

    mol.set_geom_('H 0 0 1.804; F 0 0 0', unit='B')
    mf = dft.RKS(mol)
    mf.xc = 'lda,'
    mf.conv_tol = 1e-14
    mf.kernel()
    td = tddft.TDA(mf)
    td.nstates = 3
    td.singlet = False
    e, z = td.kernel()
    tdg = Gradients(td)
    g1 = tdg.kernel(state=2)
    print(g1)
# [[ 0  0  -0.3633334]
#  [ 0  0   0.3633334]]
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[2]-e2[2])/.002 - g1[0,2]).max())

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-14
    mf.kernel()
    td = tddft.TDA(mf)
    td.nstates = 3
    td.singlet = False
    e, z = td.kernel()
    tdg = td.Gradients()
    g1 = tdg.kernel(state=2)
    print(g1)
# [[ 0  0  -0.3633334]
#  [ 0  0   0.3633334]]
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[2]-e2[2])/.002 - g1[0,2]).max())
