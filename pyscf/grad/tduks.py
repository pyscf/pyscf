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
from pyscf.dft import numint
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.scf import ucphf


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
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
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, spin=1)
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, (dmxpya,dmxpyb),
                                (dmzooa,dmzoob), True, True, max_memory)

    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
              dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if omega != 0:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        vj = vj.reshape(2,3,nao,nao)
        vk = vk.reshape(2,3,nao,nao)

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

    vresp = mf.gen_response(hermi=1, with_nlc=not td_grad.base.exclude_nlc)
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

    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
              oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vj = vj.reshape(2,4,3,nao,nao)
        vk = vk.reshape(2,4,3,nao,nao) * hyb
        if omega != 0:
            vk += td_grad.get_k(mol, dm, omega=omega).reshape(2,4,3,nao,nao) * (alpha-hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T,
              oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T)
        vj = td_grad.get_j(mol, dm).reshape(2,3,3,nao,nao)
        veff1 = numpy.zeros((2,4,3,nao,nao))
        veff1[:,:3] = vj[0] + vj[1]

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
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

        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDUKS nuclear gradients', *time0)
    return de


# dmov, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
#TODO: to include the response of grids
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmvo = [(dmvo[0] + dmvo[0].T) * .5, # because K_{ia,jb} == K_{ia,jb}
            (dmvo[1] + dmvo[1].T) * .5]

    f1vo = numpy.zeros((2,4,nao,nao))
    deriv = 2
    if dmoo is not None:
        f1oo = numpy.zeros((2,4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = numpy.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = numpy.zeros((2,4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    if mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = (ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
               ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False))
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

        rho1 = numpy.asarray((
            ni.eval_rho(mol, ao0, dmvo[0], mask, xctype, hermi=1, with_lapl=False),
            ni.eval_rho(mol, ao0, dmvo[1], mask, xctype, hermi=1, with_lapl=False)))
        if xctype == 'LDA':
            rho1 = rho1[:,numpy.newaxis]
        wv = numpy.einsum('axg,axbyg,g->byg', rho1, fxc, weight)
        fmat_(mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
        fmat_(mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None:
            rho2 = numpy.asarray((
                ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False)))
            if xctype == 'LDA':
                rho2 = rho2[:,numpy.newaxis]
            wv = numpy.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_vxc:
            wv = vxc * weight
            fmat_(mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_kxc:
            wv = numpy.einsum('axg,byg,axbyczg,g->czg', rho1, rho1, kxc, weight)
            fmat_(mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[:,1:] *= -1
    if f1oo is not None: f1oo[:,1:] *= -1
    if v1ao is not None: v1ao[:,1:] *= -1
    if k1ao is not None: k1ao[:,1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


class Gradients(tdrhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)

Grad = Gradients
