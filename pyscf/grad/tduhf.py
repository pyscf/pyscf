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
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.scf import ucphf
from pyscf import __config__


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    log = logger.new_logger(td_grad, verbose)
    time0 = time.clock(), time.time()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
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
    dmzvopa = reduce(numpy.dot, (orbva, xpya, orboa.T))
    dmzvopb = reduce(numpy.dot, (orbvb, xpyb, orbob.T))
    dmzvoma = reduce(numpy.dot, (orbva, xmya, orboa.T))
    dmzvomb = reduce(numpy.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(numpy.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(numpy.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(numpy.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(numpy.dot, (orbvb, dvvb, orbvb.T))

    vj, vk = mf.get_jk(mol, (dmzooa, dmzvopa+dmzvopa.T, dmzvoma-dmzvoma.T,
                             dmzoob, dmzvopb+dmzvopb.T, dmzvomb-dmzvomb.T), hermi=0)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
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
    def fvind(x):
        dm1 = numpy.empty((2,nao,nao))
        xa = x[0,:nvira*nocca].reshape(nvira,nocca)
        xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dma = reduce(numpy.dot, (orbva, xa, orboa.T))
        dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
        dm1[0] = dma + dma.T
        dm1[1] = dmb + dmb.T
        vj, vk = mf.get_jk(mol, dm1)
        v1 = vj[0] + vj[1] - vk
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
    vj, vk = mf.get_jk(mol, z1ao, hermi=0)
    veff = vj[0]+vj[1] - vk

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
    vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa+dmz1dooa.T, dmzvopa+dmzvopa.T, dmzvoma-dmzvoma.T,
                                  oo0b, dmz1doob+dmz1doob.T, dmzvopb+dmzvopb.T, dmzvomb-dmzvomb.T))
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
        de[k] += numpy.einsum('xij,ij->x', vhf1a[2,:,p0:p1], dmzvopa[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1b[2,:,p0:p1], dmzvopb[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1a[3,:,p0:p1], dmzvoma[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', vhf1b[3,:,p0:p1], dmzvomb[p0:p1,:])
        de[k] += numpy.einsum('xji,ij->x', vhf1a[2,:,p0:p1], dmzvopa[:,p0:p1])
        de[k] += numpy.einsum('xji,ij->x', vhf1b[2,:,p0:p1], dmzvopb[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', vhf1a[3,:,p0:p1], dmzvoma[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', vhf1b[3,:,p0:p1], dmzvomb[:,p0:p1])

    log.timer('TDUHF nuclear gradients', *time0)
    return de


class Gradients(tdrhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)

Grad = Gradients

from pyscf import tdscf
tdscf.uhf.TDA.Gradients = tdscf.uhf.TDHF.Gradients = lib.class_as_method(Gradients)


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
    mol.spin = 2
    mol.build()

    mf = scf.UHF(mol).run(conv_tol=1e-14)
    td = tddft.TDA(mf)
    td.conv_tol = 1e-14
    e, z = td.kernel()
    tdg = td.Gradients()
    #tdg.verbose = 5
    g1 = tdg.kernel(z[1])
    print(g1)
    print(lib.finger(g1) - 0.3970638627132136)
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[1]-e2[1])/.002 - g1[0,2]).max())

    mol.set_geom_('H 0 0 1.804; F 0 0 0', unit='B')
    td = tddft.TDDFT(mf)
    td.conv_tol = 1e-14
    td.nstates = 3
    e, z = td.kernel()
    tdg = Gradients(td)
    g1 = tdg.kernel(state=2)
    print(g1)
    print(lib.finger(g1) - 0.39823315202373544)
    td_solver = td.as_scanner()
    e1 = td_solver(mol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
    e2 = td_solver(mol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
    print(abs((e1[1]-e2[1])/.002 - g1[0,2]).max())

