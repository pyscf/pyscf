#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
Second order CASSCF
'''


import copy
from functools import reduce
from itertools import product
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import casci, mc1step, addons
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf import scf
from pyscf.soscf import ciah

def _pack_ci_get_H (mc, mo, ci0):
    # MRH 06/24/2020: get wrapped Hci and Hdiag functions for
    # SA and SA-mix case. Put ci in a consistent format: list of raveled CI vecs
    # so that list comprehension can do all the +-*/ and dot operations
    orbsym = getattr (mo, 'orbsym', None)
    if orbsym is None:
        orbsym = getattr (mc.fcisolver, 'orbsym', None)
    else:
        orbsym = orbsym[mc.ncore:][:mc.ncas]
    if mc.fcisolver.nroots == 1:
        ci0 = [ci0.ravel ()]
        def _pack_ci (x):
            return x[0]
        def _unpack_ci (x):
            return [x.ravel ()]
    else:
        ci0 = [c.ravel () for c in ci0]
        def _pack_ci (x):
            return numpy.concatenate ([y.ravel () for y in x])
        def _unpack_ci (x):
            z = []
            for i, c in enumerate (ci0):
                y, x = x[:c.size], x[c.size:]
                z.append (y)
            return z
        assert (len (ci0) == mc.fcisolver.nroots), '{} {}'.format (len (ci0), mc.fcisolver.nroots)
    ncas = mc.ncas
    nelecas = mc.nelecas

    # State average mix
    _state_arg = addons.StateAverageMixFCISolver_state_args
    # _solver_arg = addons.StateAverageMixFCISolver_solver_args
    if isinstance (mc.fcisolver, addons.StateAverageMixFCISolver):
        linkstrl = []
        linkstr =  []
        for solver, my_arg, my_kwargs in mc.fcisolver._loop_solver (_state_arg (ci0)):
            nelec = mc.fcisolver._get_nelec (solver, nelecas)
            if getattr (solver, 'gen_linkstr', None):
                linkstrl.append (solver.gen_linkstr (ncas, nelec, True))
                linkstr.append  (solver.gen_linkstr (ncas, nelec, False))
            else:
                linkstrl.append (None)
                linkstr.append (None)
            solver.orbsym = orbsym

        def _Hci (h1, h2, ci1):
            hci = []
            for ix, (solver, my_args, my_kwargs) in enumerate (mc.fcisolver._loop_solver (_state_arg (ci1))):
                ci = my_args[0]
                nelec = mc.fcisolver._get_nelec (solver, nelecas)
                op = solver.absorb_h1e (h1, h2, ncas, nelec, 0.5) if h1 is not None else h2
                if solver.nroots == 1: ci = [ci]
                hci.extend ((solver.contract_2e (op, c, ncas, nelec, link_index=linkstrl[ix]).ravel () for c in ci))
            return hci

        def _Hdiag (h1, h2):
            hdiag = []
            for solver, my_args, my_kwargs in mc.fcisolver._loop_solver (_state_arg (ci0)):
                nelec = mc.fcisolver._get_nelec (solver, nelecas)
                my_hdiag = solver.make_hdiag (h1, h2, ncas, nelec)
                for ix in range (solver.nroots): hdiag.append (my_hdiag)
            return hdiag

    # Not state average mix
    else:
        if getattr(mc.fcisolver, 'gen_linkstr', None):
            linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
            linkstr  = mc.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstrl = linkstr  = None

        # Should work with either state average or normal
        def _Hci (h1, h2, ket):
            op = mc.fcisolver.absorb_h1e (h1, h2, ncas, nelecas, 0.5) if h1 is not None else h2
            hci = [mc.fcisolver.contract_2e (op, k, ncas, nelecas, link_index=linkstrl).ravel () for k in ket]
            return hci

        def _Hdiag (h1, h2):
            hdiag = mc.fcisolver.make_hdiag (h1, h2, ncas, nelecas)
            return [hdiag for ix in range (len (ci0))]

    return ci0, _Hci, _Hdiag, linkstrl, linkstr, _pack_ci, _unpack_ci


# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, ci0, eris, verbose=None):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nelecas = casscf.nelecas
    nmo = mo.shape[1]
    ci0, _Hci, _Hdiag, linkstrl, linkstr, _pack_ci, _unpack_ci = _pack_ci_get_H (casscf, mo, ci0)
    weights = getattr (casscf, 'weights', [1.0]) # MRH 06/24/2020 I'm proud of myself for thinking of this :)
    nroots = len (weights)

    # MRH 06/25/2020: Unfortunately, we need gpq for each root individually in H_op below.
    # I want to minimize Python loops and I also want to call eris.ppaa as few times as possible
    # because that is disk I/O for large molecules
    # part5
    jkcaa = numpy.empty((nroots,nocc,ncas))
    # part2, part3
    vhf_a = numpy.empty((nroots,nmo,nmo))
    # part1 ~ (J + 2K)
    try:
        casdm1, casdm2 = casscf.fcisolver.states_make_rdm12 (ci0, ncas, nelecas, link_index=linkstr)
        casdm1 = numpy.asarray (casdm1)
        casdm2 = numpy.asarray (casdm2)
    except AttributeError as e:
        assert (not (isinstance (casscf.fcisolver, addons.StateAverageFCISolver))), e
        casdm1, casdm2 = casscf.fcisolver.make_rdm12 (ci0, ncas, nelecas, link_index=linkstr)
        casdm1 = numpy.asarray ([casdm1])
        casdm2 = numpy.asarray ([casdm2])
    dm2tmp = casdm2.transpose(0,2,3,1,4) + casdm2.transpose(0,1,3,2,4)
    dm2tmp = dm2tmp.reshape(nroots,ncas**2,-1)
    hdm2 = numpy.empty((nroots,nmo,ncas,nmo,ncas))
    g_dm2 = numpy.empty((nroots,nmo,ncas))
    eri_cas = numpy.empty((ncas,ncas,ncas,ncas))
    jtmp = numpy.empty ((nroots,nmo,ncas,ncas))
    ktmp = numpy.empty ((nroots,nmo,ncas,ncas))
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        if i < nocc:
            jkcaa[:,i] = numpy.einsum('ik,rik->ri', 6*kbuf[:,i]-2*jbuf[i], casdm1)
        vhf_a[:,i] = (numpy.einsum('quv,ruv->rq', jbuf, casdm1) -
                      numpy.einsum('uqv,ruv->rq', kbuf, casdm1) * .5)
        for r, (dm2, dm2t) in enumerate (zip (casdm2, dm2tmp)):
            jtmp[r] = lib.dot(jbuf.reshape(nmo,-1), dm2.reshape(ncas*ncas,-1)).reshape (nmo,ncas,ncas)
            ktmp[r] = lib.dot(kbuf.transpose(1,0,2).reshape(nmo,-1), dm2t).reshape (nmo,ncas,ncas)
        hdm2[:,i,:,:,:] = (ktmp+jtmp).transpose (0,2,1,3)
        g_dm2[:,i,:] = numpy.einsum('ruuv->rv', jtmp[:,ncore:nocc])
        if ncore <= i < nocc:
            eri_cas[i-ncore] = jbuf[ncore:nocc]
    jbuf = kbuf = jtmp = ktmp = dm2tmp = casdm2 = None
    vhf_ca = eris.vhf_c + vhf_a
    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

    ################# gradient #################
    gpq = numpy.zeros((nroots, nmo, nmo), dtype=h1e_mo.dtype)
    gpq[:,:,:ncore] = (h1e_mo[None,:,:ncore] + vhf_ca[:,:,:ncore]) * 2
    gpq[:,:,ncore:nocc] = numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc],casdm1).transpose (1,0,2)
    gpq[:,:,ncore:nocc] += g_dm2

    # Roll up intermediates that no longer need to be separated by root
    vhf_ca = numpy.einsum ('r,rpq->pq', weights, vhf_ca)
    casdm1 = numpy.einsum ('r,rpq->pq', weights, casdm1)
    jkcaa = numpy.einsum ('r,rpq->pq', weights, jkcaa)
    hdm2 = numpy.einsum ('r,rpqst->pqst', weights, hdm2)

    h1cas_0 = h1e_mo[ncore:nocc,ncore:nocc] + eris.vhf_c[ncore:nocc,ncore:nocc]
    hci0 = _Hci (h1cas_0, eri_cas, ci0)
    eci0 = [c.dot(hc) for c, hc in zip (ci0, hci0)]
    gci = [hc - c * e for hc, c, e in zip (hci0, ci0, eci0)]

    def g_update(u, fcivec):
        if not isinstance (casscf, addons.StateAverageMCSCFSolver): fcivec = [fcivec]
        uc = u[:,:ncore].copy()
        ua = u[:,ncore:nocc].copy()
        rmat = u - numpy.eye(nmo)
        ra = rmat[:,ncore:nocc].copy()
        mo1 = numpy.dot(mo, u)
        mo_c = numpy.dot(mo, uc)
        mo_a = numpy.dot(mo, ua)
        dm_c = numpy.dot(mo_c, mo_c.T) * 2

        fcivec = [c / numpy.linalg.norm (c) for c in fcivec]
        fciarg = fcivec if casscf.fcisolver.nroots > 1 else fcivec[0] # MRH 06/24/2020: this is inelegant...
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fciarg, ncas, nelecas, link_index=linkstr)
        #casscf.with_dep4 = False
        #casscf.ci_response_space = 3
        #casscf.ci_grad_trust_region = 3
        #casdm1, casdm2, gci, fcivec = casscf.update_casdm(mo, u, fcivec, 0, eris, locals())
        dm_a = reduce(numpy.dot, (mo_a, casdm1, mo_a.T))
        vj, vk = casscf.get_jk(casscf.mol, (dm_c, dm_a))
        vhf_c = reduce(numpy.dot, (mo1.T, vj[0]-vk[0]*.5, mo1[:,:nocc]))
        vhf_a = reduce(numpy.dot, (mo1.T, vj[1]-vk[1]*.5, mo1[:,:nocc]))
        h1e_mo1 = reduce(numpy.dot, (u.T, h1e_mo, u[:,:nocc]))
        p1aa = numpy.empty((nmo,ncas,ncas*ncas))
        paa1 = numpy.empty((nmo,ncas*ncas,ncas))
        aaaa = numpy.empty([ncas]*4)
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            p1aa[i] = lib.dot(ua.T, jbuf.reshape(nmo,-1))
            paa1[i] = lib.dot(kbuf.transpose(0,2,1).reshape(-1,nmo), ra)
            if ncore <= i < nocc:
                aaaa[i-ncore] = jbuf[ncore:nocc]

# active space Hamiltonian up to 2nd order
        aa11 = lib.dot(ua.T, p1aa.reshape(nmo,-1)).reshape([ncas]*4)
        aa11 = aa11 + aa11.transpose(2,3,0,1) - aaaa
        a11a = lib.dot(ra.T, paa1.reshape(nmo,-1)).reshape((ncas,)*4)
        a11a = a11a + a11a.transpose(1,0,2,3)
        a11a = a11a + a11a.transpose(0,1,3,2)
        eri_cas_2 = aa11 + a11a
        h1cas_2 = h1e_mo1[ncore:nocc,ncore:nocc] + vhf_c[ncore:nocc,ncore:nocc]
        hci0 = _Hci (h1cas_2, eri_cas_2, fcivec)
        gci = [hc - c * c.dot(hc) for c, hc in zip (fcivec, hci0)]
        gci = [g * w for g, w in zip (gci, weights)]

        g = numpy.zeros_like(h1e_mo)
        g[:,:ncore] = (h1e_mo1[:,:ncore] + vhf_c[:,:ncore] + vhf_a[:,:ncore]) * 2
        g[:,ncore:nocc] = numpy.dot(h1e_mo1[:,ncore:nocc]+vhf_c[:,ncore:nocc], casdm1)
# 0000 + 1000 + 0100 + 0010 + 0001 + 1100 + 1010 + 1001  (missing 0110 + 0101 + 0011)
        p1aa = lib.dot(u.T, p1aa.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        paa1 = lib.dot(u.T, paa1.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0,1,3,2)
        g[:,ncore:nocc] += numpy.einsum('puwx,wxuv->pv', p1aa, casdm2)
        g_orb = casscf.pack_uniq_var(g-g.T)
        gci = _pack_ci (gci)
        return numpy.hstack((g_orb*2, gci*2))

    ############## hessian, diagonal ###########

    # part7
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    h_diag = numpy.einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = numpy.einsum ('r,rpp->p', weights, gpq)
    h_diag -= g_diag + g_diag.reshape(-1,1)
    idx = numpy.arange(nmo)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal() # (pr|kl) * E(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1) * 2
    h_diag[:ncore] += v_diag * 2
    idx = numpy.arange(ncore)
    h_diag[idx,idx] -= v_diag[:ncore] * 4
    # V_{pr} E_{sq}
    tmp = numpy.einsum('ii,jj->ij', eris.vhf_c, casdm1)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -eris.vhf_c[ncore:nocc,ncore:nocc] * casdm1
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    # -2(pr|sq) + 4(pq|sr) + 4(pq|rs) - 2(ps|rq)
    tmp = 6 * eris.k_pc - 2 * eris.j_pc
    h_diag[ncore:,:ncore] += tmp[ncore:]
    h_diag[:ncore,ncore:] += tmp[ncore:].T

    # part5 and part6 diag
    # -(qr|kp) E_s^k  p in core, sk in active
    h_diag[:nocc,ncore:nocc] -= jkcaa
    h_diag[ncore:nocc,:nocc] -= jkcaa.T

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag.T
    h_diag[:,ncore:nocc] += v_diag

# Does this term contribute to internal rotation?
#    h_diag[ncore:nocc,ncore:nocc] -= v_diag[:,ncore:nocc]*2
    h_diag = casscf.pack_uniq_var(h_diag)

    hci_diag = [hd - ec - gc*c*4 for hd, ec, gc, c in zip (_Hdiag (h1cas_0, eri_cas), eci0, gci, ci0)]
    hci_diag = [h * w for h, w in zip (hci_diag, weights)]
    hdiag_all = numpy.hstack((h_diag*2, _pack_ci (hci_diag)*2))

    g_orb = numpy.einsum ('r,rpq->pq', weights, gpq)
    g_orb = casscf.pack_uniq_var(g_orb-g_orb.T)
    gci = [g * w for g, w in zip (gci, weights)]
    g_all = numpy.hstack((g_orb*2, _pack_ci (gci)*2))
    ngorb = g_orb.size

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x[:ngorb])
        ci1 = _unpack_ci (x[ngorb:])

        # H_cc
        hci1 = [hc1 - c1 * ec0 for hc1, c1, ec0 in zip (_Hci (h1cas_0, eri_cas, ci1), ci1, eci0)]
        hci1 = [hc1 - 2*(hc0 - c0*ec0)*c0.dot(c1) for hc1, hc0, c0, ec0, c1 in zip (hci1, hci0, ci0, eci0, ci1)]
        hci1 = [hc1 - 2*c0*(hc0 - c0*ec0).dot(c1) for hc1, hc0, c0, ec0, c1 in zip (hci1, hci0, ci0, eci0, ci1)]

        # H_co
        rc = x1[:,:ncore]
        ra = x1[:,ncore:nocc]
        ddm_c = numpy.zeros((nmo,nmo))
        ddm_c[:,:ncore] = rc[:,:ncore] * 2
        ddm_c[:ncore,:]+= rc[:,:ncore].T * 2
        tdm1, tdm2 = casscf.fcisolver.trans_rdm12(ci1, ci0, ncas, nelecas, link_index=linkstr)
        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1,0,3,2)
        tdm2 =(tdm2 + tdm2.transpose(2,3,0,1)) * .5
        vhf_a = numpy.empty((nmo,ncore))
        paaa = numpy.empty((nmo,ncas,ncas,ncas))
        jk = 0
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            paaa[i] = jbuf[ncore:nocc]
            vhf_a[i] = numpy.einsum('quv,uv->q', jbuf[:ncore], tdm1)
            vhf_a[i]-= numpy.einsum('uqv,uv->q', kbuf[:,:ncore], tdm1) * .5
            jk += numpy.einsum('quv,q->uv', jbuf, ddm_c[i])
            jk -= numpy.einsum('uqv,q->uv', kbuf, ddm_c[i]) * .5
        g_dm2 = numpy.einsum('puwx,wxuv->pv', paaa, tdm2)
        aaaa = numpy.dot(ra.T, paaa.reshape(nmo,-1)).reshape([ncas]*4)
        aaaa = aaaa + aaaa.transpose(1,0,2,3)
        aaaa = aaaa + aaaa.transpose(2,3,0,1)
        h1aa = numpy.dot(h1e_mo[ncore:nocc]+eris.vhf_c[ncore:nocc], ra)
        h1aa = h1aa + h1aa.T + jk
        kci0 = _Hci (h1aa, aaaa, ci0)
        kci0 = [kc0 - kc0.dot(c0) * c0 for kc0, c0 in zip (kci0, ci0)]
        hci1 = [hc1 + kc0 for hc1, kc0 in zip (hci1, kci0)]
        hci1 = [h * w for h, w in zip (hci1, weights)]

        # H_oo
        # part7
        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(lib.dot, (h1e_mo, x1, dm1))
        # part8
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        g_orb = numpy.einsum ('r,rpq->pq', weights, gpq)
        x2 -= numpy.dot((g_orb + g_orb.T), x1) * .5
        # part2
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += reduce(numpy.dot, (x1[:ncore,ncore:], vhf_ca[ncore:])) * 2
        # part3
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x1[ncore:nocc], eris.vhf_c))
        # part1
        x2[:,ncore:nocc] += numpy.einsum('purv,rv->pu', hdm2, x1[:,ncore:nocc])

        if ncore > 0:
            # part4, part5, part6
            # Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            #    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            # x2[:,:ncore] += H * x1[:,:ncore] => (becuase x1=-x1.T) =>
            # x2[:,:ncore] += -H' * x1[:ncore] => (becuase x2-x2.T) =>
            # x2[:ncore] += H' * x1[:ncore]
            va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc

        # H_oc
        s10 = [c1.dot(c0) * 2 * w for c1, c0, w in zip (ci1, ci0, weights)]
        x2[:,:ncore] += ((h1e_mo[:,:ncore]+eris.vhf_c[:,:ncore]) * sum (s10) + vhf_a) * 2
        x2[:,ncore:nocc] += numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc], tdm1)
        x2[:,ncore:nocc] += g_dm2
        x2 -= numpy.einsum ('r,rpq->pq', s10, gpq)

        # (pr<->qs)
        x2 = x2 - x2.T
        return numpy.hstack((casscf.pack_uniq_var(x2)*2, _pack_ci (hci1)*2))

    return g_all, g_update, h_op, hdiag_all

# MRH, 04/08/2019: enable multiple roots
# MRH, 06/29/2020: enable state-average mix
def extract_rotation(casscf, dr, u, ci0):
    nroots = casscf.fcisolver.nroots
    nmo = casscf.mo_coeff.shape[1]
    ngorb = numpy.count_nonzero (casscf.uniq_var_indices (nmo, casscf.ncore, casscf.ncas, casscf.frozen))
    u = numpy.dot(u, casscf.update_rotate_matrix(dr[:ngorb]))
    if nroots == 1: ci0 = [ci0]
    p0 = ngorb
    ci1 = []
    for c in ci0:
        p1 = p0 + c.size
        d = c.ravel () + dr[p0:p1]
        d *= 1./numpy.linalg.norm (d)
        ci1.append (d)
        p0 = p1
    if nroots == 1: ci1 = ci1[0]
    return u, ci1

def update_orb_ci(casscf, mo, ci0, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None:
        max_stepsize = casscf.max_stepsize

    nmo = mo.shape[1]
    # MRH, 04/08/2019: enable multiple roots
    if casscf.fcisolver.nroots == 1:
        ci0 = ci0.ravel ()
    else:
        ci0 = [c.ravel () for c in ci0]
    g_all, g_update, h_op, h_diag = gen_g_hop(casscf, mo, ci0, eris)
    ngorb = numpy.count_nonzero (casscf.uniq_var_indices (nmo, casscf.ncore, casscf.ncas, casscf.frozen))
    norm_gkf = norm_gall = numpy.linalg.norm(g_all)
    log.debug('    |g|=%5.3g (%4.3g %4.3g) (keyframe)', norm_gall,
              numpy.linalg.norm(g_all[:ngorb]),
              numpy.linalg.norm(g_all[ngorb:]))

    def precond(x, e):
        if callable(h_diag):
            x = h_diag(x, e-casscf.ah_level_shift)
        else:
            hdiagd = h_diag-(e-casscf.ah_level_shift)
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            x = x/hdiagd
        x *= 1/numpy.linalg.norm(x)
        return x

    def scale_down_step(dxi, hdxi):
        dxmax = abs(dxi).max()
        if dxmax > casscf.max_stepsize:
            scale = casscf.max_stepsize / dxmax
            log.debug1('Scale rotation by %g', scale)
            dxi *= scale
            hdxi *= scale
        return dxi, hdxi

    class Statistic:
        def __init__(self):
            self.imic = 0
            self.tot_hop = 0
            self.tot_kf = 1  # The call to gen_g_hop

    if x0_guess is None:
        x0_guess = g_all
    g_op = lambda: g_all

    stat = Statistic()
    dr = 0
    ikf = 0
    u = numpy.eye(nmo)
    ci_kf = ci0

    if norm_gall < conv_tol_grad*.3:
        return u, ci_kf, norm_gall, stat, x0_guess

    for ah_conv, ihop, w, dxi, hdxi, residual, seig \
            in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                                lindep=casscf.ah_lindep, verbose=log):
        stat.tot_hop = ihop
        norm_residual = numpy.linalg.norm(residual)
        if (ah_conv or ihop == casscf.ah_max_cycle or # make sure to use the last step
            ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or
            (seig < casscf.ah_lindep)):
            stat.imic += 1
            dxmax = abs(dxi).max()
            dxi, hdxi = scale_down_step(dxi, hdxi)

            dr += dxi
            g_all = g_all + hdxi
            norm_dr = numpy.linalg.norm(dr)
            norm_gall = numpy.linalg.norm(g_all)
            norm_gorb = numpy.linalg.norm(g_all[:ngorb])
            norm_gci = numpy.linalg.norm(g_all[ngorb:])
            log.debug('    imic %d(%d)  |g|=%3.2e (%2.1e %2.1e)  |dxi|=%3.2e '
                      'max(x)=%3.2e |dr|=%3.2e  eig=%2.1e seig=%2.1e',
                      stat.imic, ihop, norm_gall, norm_gorb, norm_gci, numpy.linalg.norm(dxi),
                      dxmax, norm_dr, w, seig)

            max_cycle = max(casscf.max_cycle_micro,
                            casscf.max_cycle_micro-int(numpy.log(norm_gkf+1e-7)*2))
            log.debug1('Set max_cycle %d', max_cycle)
            ikf += 1
            if stat.imic > 3 and norm_gall > norm_gkf*casscf.ah_grad_trust_region:
                g_all = g_all - hdxi
                dr -= dxi
                norm_gall = numpy.linalg.norm(g_all)
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (stat.imic >= max_cycle or norm_gall < conv_tol_grad*.3):
                break

            elif ((ikf >= max(casscf.kf_interval, casscf.kf_interval-numpy.log(norm_dr+1e-7)) or
                   # Insert keyframe if the keyframe and the esitimated grad are too different
                   norm_gall < norm_gkf/casscf.kf_trust_region)):
                ikf = 0
                u, ci_kf = extract_rotation(casscf, dr, u, ci_kf)
                dr[:] = 0
                g_kf1 = g_update(u, ci_kf)
                stat.tot_kf += 1
                norm_gkf1 = numpy.linalg.norm(g_kf1)
                norm_gorb = numpy.linalg.norm(g_kf1[:ngorb])
                norm_gci = numpy.linalg.norm(g_kf1[ngorb:])
                norm_dg = numpy.linalg.norm(g_kf1-g_all)
                log.debug('Adjust keyframe to |g|= %4.3g (%4.3g %4.3g) '
                          '|g-correction|= %4.3g',
                          norm_gkf1, norm_gorb, norm_gci, norm_dg)

                if (norm_dg < norm_gall*casscf.ah_grad_trust_region  # kf not too diff
                    #or norm_gkf1 < norm_gkf  # grad is decaying
                    # close to solution
                    or norm_gkf1 < conv_tol_grad*casscf.ah_grad_trust_region):
                    g_all = g_kf1
                    g_kf1 = None
                    norm_gall = norm_gkf = norm_gkf1
                else:
                    g_all = g_all - hdxi
                    dr -= dxi
                    norm_gall = norm_gkf = numpy.linalg.norm(g_all)
                    log.debug('Out of trust region. Restore previouse step')
                    break

    u, ci_kf = extract_rotation(casscf, dr, u, ci_kf)
    if casscf.fcisolver.nroots == 1:
        dci_kf = ci_kf - ci0
    else:
        dci_kf = numpy.concatenate ([(x-y).ravel () for x, y in zip (ci_kf, ci0)])
    log.debug('    tot inner=%d  |g|= %4.3g (%4.3g %4.3g) |u-1|= %4.3g  |dci|= %4.3g',
              stat.imic, norm_gall, norm_gorb, norm_gci,
              numpy.linalg.norm(u-numpy.eye(nmo)),
              numpy.linalg.norm(dci_kf))
    return u, ci_kf, norm_gkf, stat, dxi


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''Second order CASSCF driver
    '''
    log = logger.new_logger(casscf, verbose)
    log.warn('SO-CASSCF (Second order CASSCF) is an experimental feature. '
             'Its performance is bad for large systems.')

    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start SO-CASSCF (newton CASSCF)')
    if callback is None:
        callback = casscf.callback

    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, casscf.ncas, casscf.nelecas)
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    dr0 = None

    t2m = t1m = log.timer('Initializing newton CASSCF', *cput0)
    imacro = 0
    tot_hop = 0
    tot_kf  = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        u, fcivec, norm_gall, stat, dr0 = \
                update_orb_ci(casscf, mo, fcivec, eris, dr0, conv_tol_grad*.3, verbose=log)
        tot_hop += stat.tot_hop
        tot_kf  += stat.tot_kf
        t2m = log.timer('update_orb_ci', *t2m)

        eris = None
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t2m)

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and norm_gall < conv_tol_grad):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('newton CASSCF converged in %d macro (%d KF %d Hx) steps',
                 imacro, tot_kf, tot_hop)
    else:
        log.info('newton CASSCF not converged, %d macro (%d KF %d Hx) steps',
                 imacro, tot_kf, tot_hop)

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, casscf.ncas, casscf.nelecas)
    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb: # dump_chk may save casdm1
            ncas = casscf.ncas
            ncore = casscf.ncore
            nocc = ncas + ncore
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = -occ
    else:
        if casscf.natorb:
            # FIXME (pyscf-2.0): Whether to transform natural orbitals in
            # active space when this flag is enabled?
            log.warn('The attribute natorb of mcscf object affects only the '
                     'orbital canonicalization.\n'
                     'If you would like to get natural orbitals in active space '
                     'without touching core and external orbitals, an explicit '
                     'call to mc.cas_natorb_() is required')
        mo_energy = None

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('newton CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


class CASSCF(mc1step.CASSCF):
    __doc__ = casci.CASCI.__doc__ + '''CASSCF

    Extra attributes for CASSCF:

        conv_tol : float
            Converge threshold.  Default is 1e-7
        conv_tol_grad : float
            Converge threshold for CI gradients and orbital rotation gradients.
            Default is 1e-4
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is prefered.
            (see notes in max_cycle_micro_inner attribute)
            Default is 0.03.
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.
        max_cycle_micro : int
            Max number of micro (CIAH) iterations in each macro iteration.
        ah_level_shift : float, for AH solver.
            Level shift for the Davidson diagonalization in AH solver.  Default is 1e-8.
        ah_conv_tol : float, for AH solver.
            converge threshold for AH solver.  Default is 1e-12.
        ah_max_cycle : float, for AH solver.
            Max number of iterations allowd in AH solver.  Default is 30.
        ah_lindep : float, for AH solver.
            Linear dependence threshold for AH solver.  Default is 1e-14.
        ah_start_tol : flat, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 0.2.
        ah_start_cycle : int, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 2.

            ``ah_conv_tol``, ``ah_max_cycle``, ``ah_lindep``, ``ah_start_tol`` and ``ah_start_cycle``
            can affect the accuracy and performance of CASSCF solver.  Lower
            ``ah_conv_tol`` and ``ah_lindep`` might improve the accuracy of CASSCF
            optimization, but decrease the performance.

            >>> from pyscf import gto, scf, mcscf
            >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
            >>> mf = scf.UHF(mol)
            >>> mf.scf()
            >>> mc = mcscf.CASSCF(mf, 6, 6)
            >>> mc.conv_tol = 1e-10
            >>> mc.ah_conv_tol = 1e-5
            >>> mc.kernel()
            -109.044401898486001
            >>> mc.ah_conv_tol = 1e-10
            >>> mc.kernel()
            -109.044401887945668

        chkfile : str
            Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
            Default is the checkpoint file of mean field object.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        ci : ndarray
            CAS space FCI coefficients
        converged : bool
            It indicates CASSCF optimization converged or not.
        mo_coeff : ndarray
            Optimized CASSCF orbitals coefficients

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    '''
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        casci.CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen
# the max orbital rotation and CI increment, prefer small step size
        self.max_stepsize = .03
        self.max_cycle_macro = 50
        self.max_cycle_micro = 10
        self.conv_tol = 1e-7
        self.conv_tol_grad = None
        # for augmented hessian
        self.ah_level_shift = 1e-8
        self.ah_conv_tol = 1e-12
        self.ah_max_cycle = 30
        self.ah_lindep = 1e-14
        self.ah_start_tol = 5e2
        self.ah_start_cycle = 3
        self.ah_grad_trust_region = 3.

        self.kf_trust_region = 3.
        self.kf_interval = 5
        self.internal_rotation = False
        self.chkfile = self._scf.chkfile

        self.callback = None
        self.chk_ci = False

        self.fcisolver.max_cycle = 25
        #self.fcisolver.max_space = 25

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], self.ncas, ncore, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('max_cycle_micro = %d', self.max_cycle_micro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('orbital rotation max_stepsize = %g', self.max_stepsize)
        log.info('augmented hessian ah_max_cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian ah_conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian ah_linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian ah_level shift = %g', self.ah_level_shift)
        log.info('augmented hessian ah_start_tol = %g', self.ah_start_tol)
        log.info('augmented hessian ah_start_cycle = %d', self.ah_start_cycle)
        log.info('augmented hessian ah_grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_trust_region = %g', self.kf_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        if self.mo_coeff is None:
            log.warn('Orbital for CASCI is not specified.  You probably need '
                     'call SCF.kernel() to initialize orbitals.')
        return self

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        return mc1step.CASSCF.kernel(self, mo_coeff, ci0, callback, kernel)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = mc1step._fake_h_for_fast_casci(self, mo_coeff, eris)

        e_tot, e_cas, fcivec = casci.kernel(fcasci, mo_coeff, ci0, log,
                                            envs=envs)
        if not isinstance(e_cas, (float, numpy.number)):
            raise RuntimeError('Multiple roots are detected in fcisolver.  '
                               'CASSCF does not know which state to optimize.\n'
                               'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)

            if getattr(self.fcisolver, 'spin_square', None):
                try:
                    ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                except NotImplementedError:
                    ss = None
            else:
                ss = None

            if 'imacro' in envs:  # Within CASSCF iteration
                stat = envs['stat']
                if ss is None:
                    log.info('macro %d (%d JK  %d micro), '
                             'CASSCF E = %.15g  dE = %.4g  |grad|=%5.3g',
                             envs['imacro'], stat.tot_hop+stat.tot_kf, stat.imic,
                             e_tot, e_tot-envs['elast'], envs['norm_gall'])
                else:
                    log.info('macro %d (%d JK  %d micro), '
                             'CASSCF E = %.15g  dE = %.4g  |grad|=%5.3g  S^2 = %.7f',
                             envs['imacro'], stat.tot_hop+stat.tot_kf, stat.imic,
                             e_tot, e_tot-envs['elast'], envs['norm_gall'], ss[0])
            else:  # Initialization step
                elast = envs.get('elast', 0)
                if ss is None:
                    log.info('CASCI E = %.15g', e_tot)
                else:
                    log.info('CASCI E = %.15g  dE = %.8g  S^2 = %.7f',
                             e_tot, e_tot-elast, ss[0])
        return e_tot, e_cas, fcivec

    def update_ao2mo(self, mo):
        raise DeprecationWarning('update_ao2mo was obseleted since pyscf v1.0.  '
                                 'Use .ao2mo method instead')

    # Don't remove the two functions.  They are used in df/approx_hessian code
    def get_h2eff(self, mo_coeff=None):
        return self.get_h2cas(mo_coeff)
    def get_h2cas(self, mo_coeff=None):
        return casci.CASCI.ao2mo(self, mo_coeff)


if __name__ == '__main__':
    from pyscf import gto
    import pyscf.fci

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = kernel(CASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.22013929407)

    mc = CASSCF(m, 4, (3,1))
    mc.verbose = 4
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    emc = kernel(mc, m.mo_coeff, verbose=4)[1]
    print(emc - -15.950852049859-mol.energy_nuc())

    mol.atom = [
        ['H', ( 5.,-1.    , 1.   )],
        ['H', ( 0.,-5.    ,-2.   )],
        ['H', ( 4.,-0.5   ,-3.   )],
        ['H', ( 0.,-4.5   ,-1.   )],
        ['H', ( 3.,-0.5   ,-0.   )],
        ['H', ( 0.,-3.    ,-1.   )],
        ['H', ( 2.,-2.5   , 0.   )],
        ['H', ( 1., 1.    , 3.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = kernel(CASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.62638367550087, emc - -3.6268060528596635)

    mc = CASSCF(m, 4, (3,1))
    mc.verbose = 4
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    emc = kernel(mc, m.mo_coeff, verbose=4)[1]
    print(emc - -3.62638367550087)

    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASSCF(m, 6, 4)
    mc.fcisolver = pyscf.fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    #mc.analyze()
    print(emc - -75.7155632535814)

    mc.internal_rotation = True
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)
