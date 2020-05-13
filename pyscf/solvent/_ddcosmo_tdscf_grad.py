#!/usr/bin/env python
# Copyright 2020 The PySCF Developers. All Rights Reserved.
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
ddCOSMO TDA, TDHF, TDDFT gradients

The implementaitons are based on modules
pyscf.grad.tdrhf
pyscf.grad.tdrks
pyscf.grad.tduhf
pyscf.grad.tduks
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import df
from pyscf.dft import numint
from pyscf.solvent import ddcosmo
from pyscf.solvent import ddcosmo_grad
from pyscf.solvent._attach_solvent import _Solvation
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.scf import cphf, ucphf


def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        def grad_elec(self, xy, singlet, atmlst=None):
            if isinstance(self.base._scf, dft.uks.UKS):
                return tduks_grad_elec(self, xy, atmlst, self.max_memory, self.verbose)
            elif isinstance(self.base._scf, dft.rks.RKS):
                return tdrks_grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)
            elif isinstance(self.base._scf, scf.uhf.UHF):
                return tduhf_grad_elec(self, xy, atmlst, self.max_memory, self.verbose)
            elif isinstance(self.base._scf, scf.hf.RHF):
                return tdrhf_grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base._scf.make_rdm1(ao_repr=True)

            self.de_solvent = ddcosmo_grad.kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                self._write(self.mol, self.de, self.atmlst)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)


def tdrhf_grad_elec(td_grad, x_y, singlet=True, atmlst=None,
                    max_memory=2000, verbose=logger.INFO):
    '''
    See also function pyscf.grad.tdrhf.grad_elec
    '''
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

    with_solvent = getattr(td_grad.base, 'with_solvent', mf.with_solvent)

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    vj, vk = mf.get_jk(mol, (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T), hermi=0)

    if with_solvent.equilibrium_solvation:
        vj[:2] += mf.with_solvent._B_dot_x((dmzoo, dmxpy+dmxpy.T))
    else:
        vj[0] += mf.with_solvent._B_dot_x(dmzoo)

    veff0doo = vj[0] * 2 - vk[0]
    wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2

    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2

    with lib.temporary_env(mf.with_solvent, equilibrium_solvation=True):
        # set singlet=None, generate function for CPHF type response kernel
        vresp = mf.gen_response(singlet=None, hermi=1)
        def fvind(x):  # For singlet, closed shell ground state
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
    vj, vk = td_grad.get_jk(mol, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T,
                                  dmxmy-dmxmy.T))
    vj = vj.reshape(-1,3,nao,nao)
    vk = vk.reshape(-1,3,nao,nao)
    if singlet:
        vhf1 = vj * 2 - vk
    else:
        vhf1 = numpy.vstack((vj[:2]*2-vk[:2], -vk[2:]))
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += vhf1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += vhf1[0,:,p0:p1].transpose(0,2,1)
        # oo0*2 for doubly occupied orbitals
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0) * 2
        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doo)

        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', vhf1[1,:,p0:p1], oo0[p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1[2,:,p0:p1], dmxpy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1[3,:,p0:p1], dmxmy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1[2,:,p0:p1], dmxpy[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1[3,:,p0:p1], dmxmy[:,p0:p1]) * 2

    de += _grad_solvent(with_solvent, oo0*2, dmz1doo, dmxpy*2, singlet)

    log.timer('TDHF nuclear gradients', *time0)
    return de

def tdrks_grad_elec(td_grad, x_y, singlet=True, atmlst=None,
                    max_memory=2000, verbose=logger.INFO):
    '''
    See also function pyscf.grad.tdrks.grad_elec
    '''
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

    with_solvent = getattr(td_grad.base, 'with_solvent', mf.with_solvent)

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
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    dm0 = None
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
    f1vo, f1oo, vxc1, k1ao = \
            tdrks_grad._contract_xc_kernel(td_grad, mf.xc, dmxpy,
                                           dmzoo, True, True, singlet, max_memory)

    if abs(hyb) > 1e-10:
        dm = (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        if with_solvent.equilibrium_solvation:
            vj[:2] += mf.with_solvent._B_dot_x((dmzoo, dmxpy+dmxpy.T))
        else:
            vj[0] += mf.with_solvent._B_dot_x(dmzoo)

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
        vj = mf.get_j(mol, (dmzoo, dmxpy+dmxpy.T), hermi=1)
        if with_solvent.equilibrium_solvation:
            vj[:2] += mf.with_solvent._B_dot_x((dmzoo, dmxpy+dmxpy.T))
        else:
            vj[0] += mf.with_solvent._B_dot_x(dmzoo)

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

    with lib.temporary_env(mf.with_solvent, equilibrium_solvation=True):
        # set singlet=None, generate function for CPHF type response kernel
        vresp = mf.gen_response(singlet=None, hermi=1)
        def fvind(x):
            dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
            v1ao = vresp(dm+dm.T)
            return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()
        z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                        max_cycle=td_grad.cphf_max_cycle,
                        tol=td_grad.cphf_conv_tol)[0]
        z1 = z1.reshape(nvir,nocc)
        time1 = log.timer('Z-vector using CPHF solver', *time0)

        z1ao  = reduce(numpy.dot, (orbv, z1, orbo.T))
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
    if abs(hyb) > 1e-10:
        dm = (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
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
        vj = td_grad.get_j(mol, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T))
        vj = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        if singlet:
            veff1[:3] = vj * 2
        else:
            veff1[:2] = vj[:2] * 2

    fxcz1 = tdrks_grad._contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                           False, False, True, max_memory)[0]

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
        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], dmxpy[p0:p1,:]) * 2
        e1 += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], dmxmy[p0:p1,:]) * 2
        e1 += numpy.einsum('xji,ij->x', veff1[2,:,p0:p1], dmxpy[:,p0:p1]) * 2
        e1 -= numpy.einsum('xji,ij->x', veff1[3,:,p0:p1], dmxmy[:,p0:p1]) * 2

        de[k] = e1

    de += _grad_solvent(with_solvent, oo0*2, dmz1doo, dmxpy*2, singlet)

    log.timer('TDDFT nuclear gradients', *time0)
    return de

def tduhf_grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    '''
    See also function pyscf.grad.tduhf.grad_elec
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = time.clock(), time.time()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    with_solvent = getattr(td_grad.base, 'with_solvent', mf.with_solvent)

    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    (xa, xb), (ya, yb) = x_y
    xpya = (xa+ya).reshape(nocca,nvira).T
    xpyb = (xb+yb).reshape(noccb,nvirb).T
    xmya = (xa-ya).reshape(nocca,nvira).T
    xmyb = (xb-yb).reshape(noccb,nvirb).T

    dvva = numpy.einsum('ai,bi->ab', xpya, xpya) + numpy.einsum('ai,bi->ab', xmya, xmya)
    dvvb = numpy.einsum('ai,bi->ab', xpyb, xpyb) + numpy.einsum('ai,bi->ab', xmyb, xmyb)
    dooa =-numpy.einsum('ai,aj->ij', xpya, xpya) - numpy.einsum('ai,aj->ij', xmya, xmya)
    doob =-numpy.einsum('ai,aj->ij', xpyb, xpyb) - numpy.einsum('ai,aj->ij', xmyb, xmyb)
    dmxpya = reduce(numpy.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(numpy.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(numpy.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(numpy.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(numpy.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(numpy.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(numpy.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(numpy.dot, (orbvb, dvvb, orbvb.T))

    vj, vk = mf.get_jk(mol, (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                             dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T), hermi=0)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
    if with_solvent.equilibrium_solvation:
        dmxpy = dmxpya + dmxpyb
        vj[0,:2] += mf.with_solvent._B_dot_x((dmzooa+dmzoob, dmxpy+dmxpy.T))
    else:
        vj[0,0] += mf.with_solvent._B_dot_x(dmzooa+dmzoob)

    veff0doo = vj[0,0]+vj[1,0] - vk[:,0]
    wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
    wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
    veff = vj[0,1]+vj[1,1] - vk[:,1]
    veff0mopa = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0mopb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
    wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
    wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
    wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
    veff = -vk[:,2]
    veff0moma = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0momb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
    wvob -= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
    wvoa += numpy.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
    wvob += numpy.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2

    with lib.temporary_env(mf.with_solvent, equilibrium_solvation=True):
        vresp = mf.gen_response(hermi=1)
        def fvind(x):
            dm1 = numpy.empty((2,nao,nao))
            xa = x[0,:nvira*nocca].reshape(nvira,nocca)
            xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
            dma = reduce(numpy.dot, (orbva, xa, orboa.T))
            dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
            dm1[0] = dma + dma.T
            dm1[1] = dmb + dmb.T
            v1 = vresp(dm1)
            v1a = reduce(numpy.dot, (orbva.T, v1[0], orboa))
            v1b = reduce(numpy.dot, (orbvb.T, v1[1], orbob))
            return numpy.hstack((v1a.ravel(), v1b.ravel()))
        z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                               max_cycle=td_grad.cphf_max_cycle,
                               tol=td_grad.cphf_conv_tol)[0]
        time1 = log.timer('Z-vector using UCPHF solver', *time0)

        z1ao = numpy.empty((2,nao,nao))
        z1ao[0] = reduce(numpy.dot, (orbva, z1a, orboa.T))
        z1ao[1] = reduce(numpy.dot, (orbvb, z1b, orbob.T))
        veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)

    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    im0a[:nocca,:nocca] = reduce(numpy.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) * .5
    im0b[:noccb,:noccb] = reduce(numpy.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,nocca:] = numpy.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[noccb:,noccb:] = numpy.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,:nocca] = numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya)
    im0b[noccb:,:noccb] = numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb)
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya)
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb)

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = dooa * .5
    dm1b[:noccb,:noccb] = doob * .5
    dm1a[nocca:,nocca:] = dvva * .5
    dm1b[noccb:,noccb:] = dvvb * .5
    dm1a[nocca:,:nocca] = z1a * .5
    dm1b[noccb:,:noccb] = z1b * .5
    dm1a[:nocca,:nocca] += numpy.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += numpy.eye(noccb)
    im0a = reduce(numpy.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(numpy.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = z1ao[0] + dmzooa
    dmz1doob = z1ao[1] + dmzoob
    oo0a = reduce(numpy.dot, (orboa, orboa.T))
    oo0b = reduce(numpy.dot, (orbob, orbob.T))
    as_dm1 = oo0a + oo0b + (dmz1dooa + dmz1doob) * .5
    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                                  oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T))
    vj = vj.reshape(2,4,3,nao,nao)
    vk = vk.reshape(2,4,3,nao,nao)
    vhf1a, vhf1b = vj[0] + vj[1] - vk
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, as_dm1)

        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', vhf1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', vhf1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += numpy.einsum('xpq,pq->x', vhf1a[0,:,p0:p1], dmz1dooa[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,pq->x', vhf1b[0,:,p0:p1], dmz1doob[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', vhf1a[0,:,p0:p1], dmz1dooa[:,p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', vhf1b[0,:,p0:p1], dmz1doob[:,p0:p1]) * .5

        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', vhf1a[1,:,p0:p1], oo0a[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', vhf1b[1,:,p0:p1], oo0b[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', vhf1a[2,:,p0:p1], dmxpya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1b[2,:,p0:p1], dmxpyb[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1a[3,:,p0:p1], dmxmya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1b[3,:,p0:p1], dmxmyb[p0:p1,:])
        de[k] += numpy.einsum('xji,ij->x', vhf1a[2,:,p0:p1], dmxpya[:,p0:p1])
        de[k] += numpy.einsum('xji,ij->x', vhf1b[2,:,p0:p1], dmxpyb[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', vhf1a[3,:,p0:p1], dmxmya[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', vhf1b[3,:,p0:p1], dmxmyb[:,p0:p1])

    dm0 = oo0a + oo0b
    dmz1doo = (dmz1dooa + dmz1doob) * .5
    dmxpy = dmxpya + dmxpyb
    de += _grad_solvent(with_solvent, dm0, dmz1doo, dmxpy)

    log.timer('TDUHF nuclear gradients', *time0)
    return de

def tduks_grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    '''
    See also function pyscf.grad.tduks.grad_elec
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = time.clock(), time.time()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    with_solvent = getattr(td_grad.base, 'with_solvent', mf.with_solvent)

    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    (xa, xb), (ya, yb) = x_y
    xpya = (xa+ya).reshape(nocca,nvira).T
    xpyb = (xb+yb).reshape(noccb,nvirb).T
    xmya = (xa-ya).reshape(nocca,nvira).T
    xmyb = (xb-yb).reshape(noccb,nvirb).T

    dvva = numpy.einsum('ai,bi->ab', xpya, xpya) + numpy.einsum('ai,bi->ab', xmya, xmya)
    dvvb = numpy.einsum('ai,bi->ab', xpyb, xpyb) + numpy.einsum('ai,bi->ab', xmyb, xmyb)
    dooa =-numpy.einsum('ai,aj->ij', xpya, xpya) - numpy.einsum('ai,aj->ij', xmya, xmya)
    doob =-numpy.einsum('ai,aj->ij', xpyb, xpyb) - numpy.einsum('ai,aj->ij', xmyb, xmyb)
    dmxpya = reduce(numpy.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(numpy.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(numpy.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(numpy.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(numpy.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(numpy.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(numpy.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(numpy.dot, (orbvb, dvvb, orbvb.T))

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    dm0 = None
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, spin=1)
    f1vo, f1oo, vxc1, k1ao = \
            tduks_grad._contract_xc_kernel(td_grad, mf.xc, (dmxpya,dmxpyb),
                                           (dmzooa,dmzoob), True, True, max_memory)

    if abs(hyb) > 1e-10:
        dm = (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
              dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if abs(omega) > 1e-10:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        vj = vj.reshape(2,3,nao,nao)
        vk = vk.reshape(2,3,nao,nao)

        if with_solvent.equilibrium_solvation:
            dmxpy = dmxpya + dmxpyb
            vj[0,:2] += mf.with_solvent._B_dot_x((dmzooa+dmzoob, dmxpy+dmxpy.T))
        else:
            vj[0,0] += mf.with_solvent._B_dot_x(dmzooa+dmzoob)

        veff0doo = vj[0,0]+vj[1,0] - vk[:,0] + f1oo[:,0] + k1ao[:,0] * 2
        wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0,1]+vj[1,1] - vk[:,1] + f1vo[:,0] * 2
        veff0mopa = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0mopb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
        wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
        wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
        wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
        veff = -vk[:,2]
        veff0moma = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0momb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
        wvob -= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
        wvoa += numpy.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
        wvob += numpy.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2
    else:
        dm = (dmzooa, dmxpya+dmxpya.T,
              dmzoob, dmxpyb+dmxpyb.T)
        vj = mf.get_j(mol, dm, hermi=1).reshape(2,2,nao,nao)

        if with_solvent.equilibrium_solvation:
            dmxpy = dmxpya + dmxpyb
            vj[0,:2] += mf.with_solvent._B_dot_x((dmzooa+dmzoob, dmxpy+dmxpy.T))
        else:
            vj[0,0] += mf.with_solvent._B_dot_x(dmzooa+dmzoob)

        veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0] + k1ao[:,0] * 2
        wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0,1]+vj[1,1] + f1vo[:,0] * 2
        veff0mopa = reduce(numpy.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0mopb = reduce(numpy.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
        wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
        wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
        wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
        veff0moma = numpy.zeros((nmoa,nmoa))
        veff0momb = numpy.zeros((nmob,nmob))

    with lib.temporary_env(mf.with_solvent, equilibrium_solvation=True):
        vresp = mf.gen_response(hermi=1)
        def fvind(x):
            dm1 = numpy.empty((2,nao,nao))
            xa = x[0,:nvira*nocca].reshape(nvira,nocca)
            xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
            dma = reduce(numpy.dot, (orbva, xa, orboa.T))
            dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
            dm1[0] = dma + dma.T
            dm1[1] = dmb + dmb.T
            v1 = vresp(dm1)
            v1a = reduce(numpy.dot, (orbva.T, v1[0], orboa))
            v1b = reduce(numpy.dot, (orbvb.T, v1[1], orbob))
            return numpy.hstack((v1a.ravel(), v1b.ravel()))
        z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                               max_cycle=td_grad.cphf_max_cycle,
                               tol=td_grad.cphf_conv_tol)[0]
        time1 = log.timer('Z-vector using UCPHF solver', *time0)

        z1ao = numpy.empty((2,nao,nao))
        z1ao[0] = reduce(numpy.dot, (orbva, z1a, orboa.T))
        z1ao[1] = reduce(numpy.dot, (orbvb, z1b, orbob.T))
        veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)

    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    im0a[:nocca,:nocca] = reduce(numpy.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) * .5
    im0b[:noccb,:noccb] = reduce(numpy.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,nocca:] = numpy.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[noccb:,noccb:] = numpy.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,:nocca] = numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya)
    im0b[noccb:,:noccb] = numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb)
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya)
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb)

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = dooa * .5
    dm1b[:noccb,:noccb] = doob * .5
    dm1a[nocca:,nocca:] = dvva * .5
    dm1b[noccb:,noccb:] = dvvb * .5
    dm1a[nocca:,:nocca] = z1a * .5
    dm1b[noccb:,:noccb] = z1b * .5
    dm1a[:nocca,:nocca] += numpy.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += numpy.eye(noccb)
    im0a = reduce(numpy.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(numpy.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = z1ao[0] + dmzooa
    dmz1doob = z1ao[1] + dmzoob
    oo0a = reduce(numpy.dot, (orboa, orboa.T))
    oo0b = reduce(numpy.dot, (orbob, orbob.T))
    as_dm1 = oo0a + oo0b + (dmz1dooa + dmz1doob) * .5

    if abs(hyb) > 1e-10:
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
              oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vj = vj.reshape(2,4,3,nao,nao)
        vk = vk.reshape(2,4,3,nao,nao) * hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dm).reshape(2,4,3,nao,nao) * (alpha-hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T,
              oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T)
        vj = td_grad.get_j(mol, dm).reshape(2,3,3,nao,nao)
        veff1 = numpy.zeros((2,4,3,nao,nao))
        veff1[:,:3] = vj[0] + vj[1]

    fxcz1 = tduks_grad._contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                           False, False, max_memory)[0]

    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] +=(f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    veff1[:,2] += f1vo[:,1:] * 2
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        de[k] = numpy.einsum('xpq,pq->x', h1ao, as_dm1)

        de[k] += numpy.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += numpy.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1dooa[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doob[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1dooa[:,p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doob[:,p0:p1]) * .5

        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxpya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxpyb[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1a[3,:,p0:p1], dmxmya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1b[3,:,p0:p1], dmxmyb[p0:p1,:])
        de[k] += numpy.einsum('xji,ij->x', veff1a[2,:,p0:p1], dmxpya[:,p0:p1])
        de[k] += numpy.einsum('xji,ij->x', veff1b[2,:,p0:p1], dmxpyb[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxmya[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxmyb[:,p0:p1])

    dm0 = oo0a + oo0b
    dmz1doo = (dmz1dooa + dmz1doob) * .5
    dmxpy = dmxpya + dmxpyb
    de += _grad_solvent(with_solvent, dm0, dmz1doo, dmxpy)

    log.timer('TDUHF nuclear gradients', *time0)
    return de


def _grad_solvent(pcmobj, dm0, dmz1doo, dmxpy, singlet=True):
    '''Energy derivatives associated to derivatives of B tensor'''
    dielectric = pcmobj.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1

    r_vdw      = pcmobj._intermediates['r_vdw'     ]
    ylm_1sph   = pcmobj._intermediates['ylm_1sph'  ]
    ui         = pcmobj._intermediates['ui'        ]
    Lmat       = pcmobj._intermediates['Lmat'      ]
    cached_pol = pcmobj._intermediates['cached_pol']

    # First order nuclei-solvent-electron contribution
    tmp = _grad_ne(pcmobj, dmz1doo,
                   r_vdw, ui, ylm_1sph, cached_pol, Lmat)
    de = .5 * f_epsilon * tmp

    # First order electron-solvent-electron contribution
    tmp = _grad_ee(pcmobj, (dm0, dmxpy), (dmz1doo, dmxpy),
                   r_vdw, ui, ylm_1sph, cached_pol, Lmat)
    de += .5 * f_epsilon * tmp[0]  # (dm0 * dmz1doo)

    if singlet and pcmobj.equilibrium_solvation:
        de += .5 * f_epsilon * tmp[1]  # (dmxpy * dmxpy)

    return de

def _grad_nn(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''nuclei-solvent-nuclei term'''
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    de = numpy.zeros((natm,3))

    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    v_phi = numpy.zeros((natm, ngrid_1sph))
    for ia in range(natm):
# Note (-) sign is not applied to atom_charges, because (-) is explicitly
# included in rhs and L matrix
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
        v_phi[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi0 = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)

    psi0 = numpy.zeros((natm, nlm))
    for ia in range(natm):
        psi0[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)

    v_phi0 = numpy.empty((natm,ngrid_1sph))
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords
        v_phi0[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi1 = -numpy.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, v_phi0)

    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        for ja in range(natm):
            rs = atom_coords[ja] - cav_coords
            d_rs = lib.norm(rs, axis=1)
            v_phi = atom_charges[ja] * numpy.einsum('px,p->px', rs, 1./d_rs**3)
            tmp = numpy.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
            phi1[ja,:,ia] += tmp  # response of the other atoms
            phi1[ia,:,ia] -= tmp  # response of cavity grids

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)
    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0.ravel())
    Xvec0 = Xvec0.reshape(natm,nlm)
    phi1 -= numpy.einsum('aziljm,jm->azil', L1, Xvec0)

    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0.ravel())
    LS0 = LS0.reshape(natm,nlm)
    de += numpy.einsum('il,azil->az', LS0, phi1)

    return de

def _grad_ne(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''nuclear charge-electron density cross term'''
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2
    nao = mol.nao
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids
    aoslices = mol.aoslice_by_atom()

    #extern_point_idx = ui > 0
    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    dms = numpy.asarray(dm)
    is_single_dm = dms.ndim == 2
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    de = numpy.zeros((n_dm,natm,3))

    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    v_phi = numpy.zeros((natm, ngrid_1sph))
    for ia in range(natm):
# Note (-) sign is not applied to atom_charges, because (-) is explicitly
# included in rhs and L matrix
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
        v_phi[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi0 = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0.ravel())
    Xvec0 = Xvec0.reshape(natm,nlm)

    v_phi0 = numpy.empty((natm,ngrid_1sph))
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords
        v_phi0[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi1 = -numpy.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, v_phi0)

    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        for ja in range(natm):
            rs = atom_coords[ja] - cav_coords
            d_rs = lib.norm(rs, axis=1)
            v_phi = atom_charges[ja] * numpy.einsum('px,p->px', rs, 1./d_rs**3)
            tmp = numpy.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
            phi1[ja,:,ia] += tmp  # response of the other atoms
            phi1[ia,:,ia] -= tmp  # response of cavity grids

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)

    phi1 -= numpy.einsum('aziljm,jm->azil', L1, Xvec0)
    Xvec1 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi1.reshape(-1,natm*nlm).T)
    Xvec1 = Xvec1.T.reshape(natm,3,natm,nlm)

    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        ao = mol.eval_gto('GTOval_sph_deriv1', coords)

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        scaled_weights = numpy.einsum('azm,mn->azn', Xvec1[:,:,ia], fac_pol)
        scaled_weights *= weight
        aow = numpy.einsum('gi,azg->azgi', ao[0], scaled_weights)
        de -= numpy.einsum('nij,gi,azgj->naz', dms, ao[0], aow)

        aow0 = numpy.einsum('gi,g->gi', ao[0], weight)
        aow1 = numpy.einsum('gi,zxg->zxgi', ao[0], weight1)

        den0 = numpy.einsum('nij,gi,zxgj->nzxg', dms, ao[0], aow1)
        de -= numpy.einsum('m,mg,nzxg->nzx', Xvec0[ia], fac_pol, den0)

        eta_nj = numpy.einsum('m,mg->g', Xvec0[ia], fac_pol)
        dm_ao  = lib.einsum('nij,gj->ngi', dms, aow0)
        dm_ao += lib.einsum('nji,gj->ngi', dms, aow0)
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            den1 = numpy.einsum('ngi,xgi->nxg', dm_ao[:,:,p0:p1], ao[1:,:,p0:p1])
            detmp = numpy.einsum('g,nxg->nx', eta_nj, den1)
            de[:,ja] += detmp
            de[:,ia] -= detmp

    psi0 = numpy.zeros((natm, nlm))
    for ia in range(natm):
        psi0[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)

    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0.ravel())
    LS0 = LS0.reshape(natm,nlm)

    LS1 = numpy.einsum('il,aziljm->azjm', LS0, L1)
    LS1 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, LS1.reshape(-1,natm*nlm).T)
    LS1 = LS1.T.reshape(natm,3,natm,nlm)

    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    cintopt_ip1 = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        wtmp = numpy.einsum('l,n,ln->ln', LS0[ia], weights_1sph, ylm_1sph)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        jaux = numpy.einsum('ijg,nij->ng', v_nj, dms)
        de -= numpy.einsum('azl,g,lg,g,ng->naz', LS1[:,:,ia], weights_1sph, ylm_1sph, ui[ia], jaux)
        de += numpy.einsum('lg,azg,ng->naz', wtmp, ui1[:,:,ia], jaux)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3,
                                   aosym='s1', cintopt=cintopt_ip1)
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            jaux1  = numpy.einsum('xijg,nij->nxg', v_e1_nj[:,p0:p1], dms[:,p0:p1])
            jaux1 += numpy.einsum('xijg,nji->nxg', v_e1_nj[:,p0:p1], dms[:,:,p0:p1])
            detmp = numpy.einsum('lg,g,nxg->nx', wtmp, ui[ia], jaux1)
            de[:,ja] -= detmp
            de[:,ia] += detmp

    if is_single_dm:
        de = de[0]
    return de

def _grad_ee(pcmobj, dm1, dm2, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''electron density-electorn density term'''
    mol = pcmobj.mol
    mol = pcmobj.mol
    natm = mol.natm
    nao  = mol.nao
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    atom_coords = mol.atom_coords()
    aoslices = mol.aoslice_by_atom()
    grids = pcmobj.grids
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)

    #extern_point_idx = ui > 0
    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    dm1s = numpy.asarray(dm1)
    dm2s = numpy.asarray(dm2)
    is_single_dm = dm1s.ndim == 2
    dm1s = dm1s.reshape(-1,nao,nao)
    dm2s = dm2s.reshape(-1,nao,nao)
    n_dm = dm1s.shape[0]
    assert dm2s.shape[0] == n_dm

    de = numpy.zeros((n_dm,natm,3))

    ni = numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm1s, hermi=0)
    den = numpy.empty((n_dm,grids.weights.size))
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 0):
        p0, p1 = p1, p1 + weight.size
        for i in range(n_dm):
            den[i,p0:p1] = make_rho(i, ao, mask, 'LDA') * weight

    psi0_dm1 = numpy.zeros((n_dm, natm, nlm))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        i0, i1 = i1, i1 + fac_pol.shape[1]
        psi0_dm1[:,ia] = -numpy.einsum('mg,ng->nm', fac_pol, den[:,i0:i1])
    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0_dm1.reshape(n_dm,-1).T)
    LS0 = LS0.T.reshape(n_dm,natm,nlm)

    phi0_dm1 = numpy.zeros((n_dm,natm,nlm))
    phi0_dm2 = numpy.zeros((n_dm,natm,nlm))
    phi1_dm1 = numpy.zeros((n_dm,natm,3,natm,nlm))
    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    cintopt_ip1 = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)

        v_phi = numpy.einsum('ijg,nij->ng', v_nj, dm1s)
        phi0_dm1[:,ia] = numpy.einsum('g,lg,g,ng->nl', weights_1sph, ylm_1sph, ui[ia], v_phi)
        phi1_dm1[:,:,:,ia] += numpy.einsum('g,lg,azg,ng->nazl', weights_1sph, ylm_1sph, ui1[:,:,ia], v_phi)

        jaux = numpy.einsum('ijg,nij->ng', v_nj, dm2s)
        de += numpy.einsum('nl,g,lg,azg,ng->naz', LS0[:,ia], weights_1sph, ylm_1sph, ui1[:,:,ia], jaux)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3,
                                   aosym='s1', cintopt=cintopt_ip1)
        wtmp = numpy.einsum('g,lg,g->lg', weights_1sph, ylm_1sph, ui[ia])
        phi0_dm2[:,ia] = numpy.einsum('lg,ng->nl', wtmp, jaux)
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            jaux1  = numpy.einsum('xijg,nij->nxg', v_e1_nj[:,p0:p1], dm2s[:,p0:p1])
            jaux1 += numpy.einsum('xijg,nji->nxg', v_e1_nj[:,p0:p1], dm2s[:,:,p0:p1])
            detmp = numpy.einsum('nl,lg,nxg->nx', LS0[:,ia], wtmp, jaux1)
            de[:,ja] -= detmp
            de[:,ia] += detmp

            tmp  = numpy.einsum('xijg,nij->nxg', v_e1_nj[:,p0:p1], dm1s[:,p0:p1])
            tmp += numpy.einsum('xijg,nji->nxg', v_e1_nj[:,p0:p1], dm1s[:,:,p0:p1])
            phitmp = numpy.einsum('lg,nxg->nxl', wtmp, tmp)
            phi1_dm1[:,ja,:,ia] -= phitmp
            phi1_dm1[:,ia,:,ia] += phitmp

    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0_dm1.reshape(n_dm,-1).T)
    Xvec0 = Xvec0.T.reshape(n_dm,natm,nlm)

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)

    phi1_dm1 -= numpy.einsum('aziljm,njm->nazil', L1, Xvec0)
    Xvec1 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi1_dm1.reshape(-1,natm*nlm).T)
    Xvec1 = Xvec1.T.reshape(n_dm,natm,3,natm,nlm)

    psi1_dm1 = numpy.zeros((n_dm,natm,3,natm,nlm))
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        ao = mol.eval_gto('GTOval_sph_deriv1', coords)

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        scaled_weights = numpy.einsum('nazm,mg->nazg', Xvec1[:,:,:,ia], fac_pol)
        scaled_weights *= weight
        aow = numpy.einsum('gi,nazg->nazgi', ao[0], scaled_weights)
        de -= numpy.einsum('nij,gi,nazgj->naz', dm2s, ao[0], aow)

        aow0 = numpy.einsum('gi,g->gi', ao[0], weight)
        aow1 = numpy.einsum('gi,zxg->zxgi', ao[0], weight1)

        den0 = numpy.einsum('nij,gi,zxgj->nzxg', dm1s, ao[0], aow1)
        psi1_dm1[:,:,:,ia] -= numpy.einsum('mg,nzxg->nzxm', fac_pol, den0)

        dm_ao  = lib.einsum('nij,gj->ngi', dm1s, aow0)
        dm_ao += lib.einsum('nji,gj->ngi', dm1s, aow0)
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            den1 = numpy.einsum('ngi,xgi->nxg', dm_ao[:,:,p0:p1], ao[1:,:,p0:p1])
            psitmp = numpy.einsum('mg,nxg->nxm', fac_pol, den1)
            psi1_dm1[:,ja,:,ia] += psitmp
            psi1_dm1[:,ia,:,ia] -= psitmp

        eta_nj = numpy.einsum('nm,mg->ng', Xvec0[:,ia], fac_pol)
        den0 = numpy.einsum('nij,gi,zxgj->nzxg', dm2s, ao[0], aow1)
        de -= numpy.einsum('ng,nzxg->nzx', eta_nj, den0)

        dm_ao  = lib.einsum('nij,gj->ngi', dm2s, aow0)
        dm_ao += lib.einsum('nji,gj->ngi', dm2s, aow0)
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            den1 = lib.einsum('ngi,xgi->nxg', dm_ao[:,:,p0:p1], ao[1:,:,p0:p1])
            detmp = numpy.einsum('ng,nxg->nx', eta_nj, den1)
            de[:,ja] += detmp
            de[:,ia] -= detmp

    psi1_dm1 -= numpy.einsum('nil,aziljm->nazjm', LS0, L1)
    LS1 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi1_dm1.reshape(-1,natm*nlm).T)
    LS1 = LS1.T.reshape(n_dm,natm,3,natm,nlm)
    de += numpy.einsum('nazjx,njx->naz', LS1, phi0_dm2)

    if is_single_dm:
        de = de[0]
    return de


if __name__ == '__main__':
    mol0 = gto.M(atom='H  0.  0.  1.804; F  0.  0.  0.', verbose=0, unit='B')
    mol1 = gto.M(atom='H  0.  0.  1.803; F  0.  0.  0.', verbose=0, unit='B')
    mol2 = gto.M(atom='H  0.  0.  1.805; F  0.  0.  0.', verbose=0, unit='B')

    # TDA with equilibrium_solvation
    mf = mol0.RHF().ddCOSMO().run()
    td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    g1 = td.nuc_grad_method().kernel() # 0  0  -0.5116214042

    mf = mol1.RHF().ddCOSMO().run()
    td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    mf = mol2.RHF().ddCOSMO().run()
    td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    print((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2])
    print((td2.e_tot[0]-td1.e_tot[0])/0.002 - g1[0,2])

    # TDA without equilibrium_solvation
    mf = mol0.RHF().ddCOSMO().run()
    td = mf.TDA().ddCOSMO().run()
    g1 = td.nuc_grad_method().kernel()

    mf = mol1.RHF().ddCOSMO().run()
    td1 = mf.TDA().ddCOSMO().run()
    mf = mol2.RHF().ddCOSMO().run()
    td2 = mf.TDA().ddCOSMO().run()
    print((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2])
    print((td2.e_tot[0]-td1.e_tot[0])/0.002 - g1[0,2])

    # TDA lda with equilibrium_solvation
    mf = mol0.RKS().ddCOSMO().run(xc='svwn')
    td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    g1 = td.nuc_grad_method().kernel()

    mf = mol1.RKS().ddCOSMO().run(xc='svwn')
    td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    mf = mol2.RKS().ddCOSMO().run(xc='svwn')
    td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
    print((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2])
    print((td2.e_tot[0]-td1.e_tot[0])/0.002 - g1[0,2])
