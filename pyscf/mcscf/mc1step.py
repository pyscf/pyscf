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

import sys

from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import casci
from pyscf.mcscf.casci import CASCI, get_fock, cas_natorb, canonicalize
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile
from pyscf import ao2mo
from pyscf import scf
from pyscf.soscf import ciah
from pyscf import __config__

WITH_MICRO_SCHEDULER = getattr(__config__, 'mcscf_mc1step_CASSCF_with_micro_scheduler', False)
WITH_STEPSIZE_SCHEDULER = getattr(__config__, 'mcscf_mc1step_CASSCF_with_stepsize_scheduler', True)

# ref. JCP, 82, 5053 (1985); DOI: 10.1063/1.448627 and JCP 73, 2342 (1980); DOI:10.1063/1.440384

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, u, casdm1, casdm2, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    # part5
    jkcaa = numpy.empty((nocc,ncas))
    # part2, part3
    vhf_a = numpy.empty((nmo,nmo))
    # part1 ~ (J + 2K)
    dm2tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3)
    dm2tmp = dm2tmp.reshape(ncas**2,-1)
    hdm2 = numpy.empty((nmo,ncas,nmo,ncas))
    g_dm2 = numpy.empty((nmo,ncas))
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        if i < nocc:
            jkcaa[i] = numpy.einsum('ik,ik->i', 6*kbuf[:,i]-2*jbuf[i], casdm1)
        vhf_a[i] =(numpy.einsum('quv,uv->q', jbuf, casdm1) -
                   numpy.einsum('uqv,uv->q', kbuf, casdm1) * .5)
        jtmp = lib.dot(jbuf.reshape(nmo,-1), casdm2.reshape(ncas*ncas,-1))
        jtmp = jtmp.reshape(nmo,ncas,ncas)
        ktmp = lib.dot(kbuf.transpose(1,0,2).reshape(nmo,-1), dm2tmp)
        hdm2[i] = (ktmp.reshape(nmo,ncas,ncas)+jtmp).transpose(1,0,2)
        g_dm2[i] = numpy.einsum('uuv->v', jtmp[ncore:nocc])
    jbuf = kbuf = jtmp = ktmp = dm2tmp = None
    vhf_ca = eris.vhf_c + vhf_a
    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

    ################# gradient #################
    g = numpy.zeros_like(h1e_mo)
    g[:,:ncore] = (h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
    g[:,ncore:nocc] = numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc],casdm1)
    g[:,ncore:nocc] += g_dm2

    def gorb_update(u, fcivec):
        uc = u[:,:ncore].copy()
        ua = u[:,ncore:nocc].copy()
        rmat = u - numpy.eye(nmo)
        ra = rmat[:,ncore:nocc].copy()
        mo1 = numpy.dot(mo, u)
        mo_c = numpy.dot(mo, uc)
        mo_a = numpy.dot(mo, ua)
        dm_c = numpy.dot(mo_c, mo_c.T) * 2

        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas)
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

        g = numpy.zeros_like(h1e_mo)
        g[:,:ncore] = (h1e_mo1[:,:ncore] + vhf_c[:,:ncore] + vhf_a[:,:ncore]) * 2
        g[:,ncore:nocc] = numpy.dot(h1e_mo1[:,ncore:nocc]+vhf_c[:,ncore:nocc], casdm1)
# 0000 + 1000 + 0100 + 0010 + 0001 + 1100 + 1010 + 1001  (missing 0110 + 0101 + 0011)
        p1aa = lib.dot(u.T, p1aa.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        paa1 = lib.dot(u.T, paa1.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0,1,3,2)
        g[:,ncore:nocc] += numpy.einsum('puwx,wxuv->pv', p1aa, casdm2)
        return casscf.pack_uniq_var(g-g.T)

    ############## hessian, diagonal ###########

    # part7
    h_diag = numpy.einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = g.diagonal()
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

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)

        # part7
        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(lib.dot, (h1e_mo, x1, dm1))
        # part8
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        x2 -= numpy.dot((g+g.T), x1) * .5
        # part2
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += reduce(numpy.dot, (x1[:ncore,ncore:], vhf_ca[ncore:])) * 2
        # part3
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x1[ncore:nocc], eris.vhf_c))
        # part1
        x2[:,ncore:nocc] += numpy.einsum('purv,rv->pu', hdm2, x1[:,ncore:nocc])

        # part4, part5, part6
        if ncore > 0:
            # Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            #    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
            # x2[:,:ncore] += H * x1[:,:ncore] => (because x1=-x1.T) =>
            # x2[:,:ncore] += -H' * x1[:ncore] => (because x2-x2.T) =>
            # x2[:ncore] += H' * x1[:ncore]
            va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc

        # (pr<->qs)
        x2 = x2 - x2.T
        return casscf.pack_uniq_var(x2)

    return g_orb, gorb_update, h_op, h_diag

def rotate_orb_cc(casscf, mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None:
        max_stepsize = casscf.max_stepsize

    t3m = (logger.process_clock(), logger.perf_counter())
    u = 1
    g_orb, gorb_update, h_op, h_diag = \
            casscf.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)
    g_kf = g_orb
    norm_gkf = norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|=%5.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t3m)

    if norm_gorb < conv_tol_grad*.3:
        u = casscf.update_rotate_matrix(g_orb*0)
        yield u, g_orb, 1, x0_guess
        return

    def precond(x, e):
        hdiagd = h_diag-(e-casscf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        norm_x = numpy.linalg.norm(x)
        x *= 1/norm_x
        #if norm_x < 1e-2:
        #    x *= 1e-2/norm_x
        return x

    jkcount = 0
    if x0_guess is None:
        x0_guess = g_orb
    imic = 0
    dr = 0
    ikf = 0
    g_op = lambda: g_orb
    problem_size = g_orb.size

    for ah_end, ihop, w, dxi, hdxi, residual, seig \
            in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                                lindep=casscf.ah_lindep, verbose=log):
        # residual = v[0] * (g+(h-e)x) ~ v[0] * grad
        norm_residual = numpy.linalg.norm(residual)
        if (ah_end or ihop == casscf.ah_max_cycle or # make sure to use the last step
            ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or
            (seig < casscf.ah_lindep)):
            imic += 1
            dxmax = numpy.max(abs(dxi))
            if ihop == problem_size:
                log.debug1('... Hx=g fully converged for small systems')
                #max_stepsize = casscf.max_stepsize * 10
            elif dxmax > max_stepsize:
                scale = max_stepsize / dxmax
                log.debug1('... scale rotation size %g', scale)
                dxi *= scale
                hdxi *= scale

            g_orb = g_orb + hdxi
            dr = dr + dxi
            norm_gorb = numpy.linalg.norm(g_orb)
            norm_dxi = numpy.linalg.norm(dxi)
            norm_dr = numpy.linalg.norm(dr)
            log.debug('    imic %2d(%2d)  |g[o]|=%5.3g  |dxi|=%5.3g  '
                      'max(|x|)=%5.3g  |dr|=%5.3g  eig=%5.3g  seig=%5.3g',
                      imic, ihop, norm_gorb, norm_dxi,
                      dxmax, norm_dr, w, seig)

            ikf += 1
            if ikf > 1 and norm_gorb > norm_gkf*casscf.ah_grad_trust_region:
                g_orb = g_orb - hdxi
                dr -= dxi
                #norm_gorb = numpy.linalg.norm(g_orb)
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (norm_gorb < conv_tol_grad*.3):
                break

            elif (ikf >= max(casscf.kf_interval, -numpy.log(norm_dr+1e-7)) or
                  # Insert keyframe if the keyframe and the estimated grad
                  # are very different
                  norm_gorb < norm_gkf/casscf.kf_trust_region):
                ikf = 0
                u = casscf.update_rotate_matrix(dr, u)
                t3m = log.timer('aug_hess in %2d inner iters' % imic, *t3m)
                yield u, g_kf, ihop+jkcount, dxi

                t3m = (logger.process_clock(), logger.perf_counter())
# TODO: test whether to update h_op, h_diag to change the orbital hessian.
# It leads to the different hessian operations in the same davidson
# diagonalization procedure.  This is generally a bad approximation because it
# results in ill-defined hessian eigenvalue in the davidson algorithm.  But in
# certain cases, it is a small perturbation that help the mcscf optimization
# algorithm move out of local minimum
#                h_op, h_diag = \
#                        casscf.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)[2:4]
                g_kf1 = gorb_update(u, fcivec())
                jkcount += 1

                norm_gkf1 = numpy.linalg.norm(g_kf1)
                norm_dg = numpy.linalg.norm(g_kf1-g_orb)
                log.debug('    |g|=%5.3g (keyframe), |g-correction|=%5.3g',
                          norm_gkf1, norm_dg)
#
# Special treatment if out of trust region
#
                if (norm_dg > norm_gorb*casscf.ah_grad_trust_region and
                    norm_gkf1 > norm_gkf and
                    norm_gkf1 > norm_gkf*casscf.ah_grad_trust_region):
                    log.debug('    Keyframe |g|=%5.3g  |g_last| =%5.3g out of trust region',
                              norm_gkf1, norm_gorb)
# Slightly moving forward, not completely restoring last step.
# In some cases, the optimization moves out of trust region in the first micro
# iteration.  The small forward step can ensure the orbital changes in the
# current iteration.
                    dr = -dxi * (1 - casscf.scale_restoration)
                    g_kf = g_kf1
                    break
                t3m = log.timer('gen h_op', *t3m)
                g_orb = g_kf = g_kf1
                norm_gorb = norm_gkf = norm_gkf1
                dr[:] = 0

    u = casscf.update_rotate_matrix(dr, u)
    yield u, g_kf, ihop+jkcount, dxi


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''quasi-newton CASSCF optimization driver
    '''
    from pyscf.mcscf.addons import StateAverageMCSCFSolver
    log = logger.new_logger(casscf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 1-step CASSCF')
    if callback is None:
        callback = casscf.callback

    if ci0 is None:
        ci0 = casscf.ci

    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas

    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
# macro iterations are needed when added solvent model
#    if ncas == nmo and not casscf.internal_rotation:
#        if casscf.canonicalization:
#            log.debug('CASSCF canonicalization')
#            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
#                                                        casscf.sorting_mo_energy,
#                                                        casscf.natorb, verbose=log)
#        else:
#            mo_energy = None
#        return True, e_tot, e_cas, fcivec, mo, mo_energy

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    r0 = None

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    imacro = 0
    dr0 = None
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        imicro = 0
        rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad*.3, max_stepsize, log)
        for u, g_orb, njk, r0 in rota:
            imicro += 1
            norm_gorb = numpy.linalg.norm(g_orb)
            if imicro == 1:
                norm_gorb0 = norm_gorb
            norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
            t3m = log.timer('orbital rotation', *t3m)
            if imicro >= max_cycle_micro:
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g',
                          imicro, norm_t, norm_gorb)
                break

            casdm1, casdm2, gci, fcivec = \
                    casscf.update_casdm(mo, u, fcivec, e_cas, eris, locals())
            norm_ddm = numpy.linalg.norm(casdm1 - casdm1_last)
            norm_ddm_micro = numpy.linalg.norm(casdm1 - casdm1_prev)
            casdm1_prev = casdm1
            t3m = log.timer('update CAS DM', *t3m)
            if isinstance(gci, numpy.ndarray):
                norm_gci = numpy.linalg.norm(gci)
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%5.3g  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
            else:
                norm_gci = None
                log.debug('micro %2d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%s  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)

            if callable(callback):
                callback(locals())

            t3m = log.timer('micro iter %2d'%imicro, *t3m)
            if (norm_t < conv_tol_grad or
                (norm_gorb < conv_tol_grad*.5 and
                 (norm_ddm < conv_tol_ddm*.4 or norm_ddm_micro < conv_tol_ddm*.4))):
                break

        rota.close()
        rota = None

        totmicro += imicro
        totinner += njk

        eris = None
        # keep u, g_orb in locals() so that they can be accessed by callback
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        max_offdiag_u = numpy.abs(numpy.triu(u, 1)).max()
        if max_offdiag_u < casscf.small_rot_tol:
            small_rot = True
        else:
            small_rot = False
        if not isinstance(casscf, StateAverageMCSCFSolver):
            # The fcivec from builtin FCI solver is a numpy.ndarray
            if not isinstance(fcivec, numpy.ndarray):
                fcivec = small_rot
        else:
            newvecs = []
            for subvec in fcivec:
                # CI vector obtained by builtin FCI is a numpy array
                if not isinstance(subvec, numpy.ndarray):
                    newvecs.append(small_rot)
                else:
                    newvecs.append(subvec)
            fcivec = newvecs

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_last)
        casdm1_prev = casdm1_last = casdm1
        log.timer('CASCI solver', *t2m)
        t3m = t2m = t1m = log.timer('macro iter %2d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and norm_gorb0 < conv_tol_grad and
                norm_ddm < conv_tol_ddm and
                (max_offdiag_u < casscf.small_rot_tol or casscf.small_rot_tol == 0)):
            conv = True

        if dump_chk and casscf.chkfile:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('1-step CASSCF converged in %3d macro (%3d JK %3d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %3d macro (%3d JK %3d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = numpy.diag(-occ)
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

    if dump_chk and casscf.chkfile:
        casscf.dump_chk(locals())

    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


def as_scanner(mc):
    '''Generating a scanner for CASSCF PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CASSCF energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of MCSCF object
    (conv_tol, max_memory etc) are automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.2', verbose=0)
    >>> mc_scanner = mcscf.CASSCF(scf.RHF(mol), 4, 4).as_scanner()
    >>> e = mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> e = mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    if isinstance(mc, lib.SinglePointScanner):
        return mc

    logger.info(mc, 'Create scanner for %s', mc.__class__)
    name = mc.__class__.__name__ + CASSCF_Scanner.__name_mixin__
    return lib.set_class(CASSCF_Scanner(mc), (CASSCF_Scanner, mc.__class__), name)

class CASSCF_Scanner(lib.SinglePointScanner):
    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        self._scf = mc._scf.as_scanner()

    def __call__(self, mol_or_geom, mo_coeff=None, ci0=None):
        from pyscf.mcscf.addons import project_init_guess
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # These properties can be updated when calling mf_scanner(mol) if
        # they are shared with mc._scf. In certain scenario the properties
        # may be created for mc separately, e.g. when mcscf.approx_hessian is
        # called. For safety, the code below explicitly resets these
        # properties.
        self.reset (mol)
        for key in ('with_df', 'with_x2c', 'with_solvent', 'with_dftd3'):
            sub_mod = getattr(self, key, None)
            if sub_mod:
                sub_mod.reset(mol)

        mf_scanner = self._scf
        mf_scanner(mol)
        self.mol = mol

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            mo_coeff = mf_scanner.mo_coeff
        else:
            mo_coeff = project_init_guess(self, mo_coeff)
        if ci0 is None:
            ci0 = self.ci
        e_tot = self.kernel(mo_coeff, ci0)[0]
        return e_tot

def max_stepsize_scheduler(casscf, envs):
    if not WITH_STEPSIZE_SCHEDULER:
        return casscf.max_stepsize

    _max_stepsize = envs.get ('max_stepsize', None)
    if _max_stepsize is None:
        _max_stepsize = casscf.max_stepsize
    if envs['de'] > -casscf.conv_tol:  # Avoid total energy increasing
        _max_stepsize *= .3
        logger.debug(casscf, 'set max_stepsize to %g', _max_stepsize)
    else:
        _max_stepsize = (casscf.max_stepsize*_max_stepsize)**.5
    casscf._max_stepsize = _max_stepsize # for inspection by user
    return _max_stepsize

# To extend CASSCF for certain CAS space solver, it can be done by assign an
# object or a module to CASSCF.fcisolver.  The fcisolver object or module
# should at least have three member functions "kernel" (wfn for given
# hamiltonian), "make_rdm12" (1- and 2-pdm), "absorb_h1e" (effective
# 2e-hamiltonian) in 1-step CASSCF solver, and two member functions "kernel"
# and "make_rdm12" in 2-step CASSCF solver
class CASSCF(casci.CASBase):
    __doc__ = casci.CASBase.__doc__ + '''

    Extra attributes for CASSCF:

        conv_tol : float
            Converge threshold.  Default is 1e-7
        conv_tol_grad : float
            Converge threshold for CI gradients and orbital rotation gradients.
            If not specified, it is set to sqrt(conv_tol).
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is prefered.
            Default is 0.02.
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.
        max_cycle_micro : int
            Max number of micro iterations in each macro iteration.  Depending on
            systems, increasing this value might reduce the total macro
            iterations.  Generally, 2 - 5 steps should be enough.  Default is 4.
        small_rot_tol : float
            Threshold for orbital rotation to be considered small. If the largest orbital
            rotation is smaller than this value, the CI solver will restart from the
            previous iteration if supported.
            Default is 0.01
        ah_level_shift : float, for AH solver.
            Level shift for the Davidson diagonalization in AH solver.  Default is 1e-8.
        ah_conv_tol : float, for AH solver.
            converge threshold for AH solver.  Default is 1e-12.
        ah_max_cycle : float, for AH solver.
            Max number of iterations allowd in AH solver.  Default is 30.
        ah_lindep : float, for AH solver.
            Linear dependence threshold for AH solver.  Default is 1e-14.
        ah_start_tol : float, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 2.5.
        ah_start_cycle : int, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 3.

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
            >>> mc.kernel()[0]
            -109.044401898486001
            >>> mc.ah_conv_tol = 1e-10
            >>> mc.kernel()[0]
            -109.044401887945668

        chkfile : str
            Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
            Default is the checkpoint file of mean field object.
        ci_response_space : int
            subspace size to solve the CI vector response.  Default is 3.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
        scale_restoration : float
            When a step of orbital rotation moves out of trust region, the
            orbital optimization will be restored to previous state and the
            step size of the orbital rotation needs to be reduced.
            scale_restoration controls how much to scale down the step size.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray
            Optimized CASSCF orbitals coefficients. When canonicalization is
            specified, the returned orbitals make the general Fock matrix
            (Fock operator on top of MCSCF 1-particle density matrix)
            diagonalized within each subspace (core, active, external). If
            natorb (natural orbitals in active space) is enabled, the active
            segment of mo_coeff is transformed to natural orbitals.
        mo_energy : ndarray
            Diagonal elements of general Fock matrix (in mo_coeff
            representation).

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    '''

# the max orbital rotation and CI increment, prefer small step size
    max_stepsize = getattr(__config__, 'mcscf_mc1step_CASSCF_max_stepsize', .02)
    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol_grad', None)
    # for augmented hessian
    ah_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
# * ah_start_tol and ah_start_cycle control the start point to use AH step.
#   In function rotate_orb_cc, the orbital rotation is carried out with the
#   approximate aug_hessian step after a few davidson updates of the AH eigen
#   problem.  Reducing ah_start_tol or increasing ah_start_cycle will delay
#   the start point of orbital rotation.
# * We can do early ah_start since it only affect the first few iterations.
#   The start tol will be reduced when approach the convergence point.
# * Be careful with the SYMMETRY BROKEN caused by ah_start_tol/ah_start_cycle.
#   ah_start_tol/ah_start_cycle actually approximates the hessian to reduce
#   the J/K evaluation required by AH.  When the system symmetry is higher
#   than the one given by mol.symmetry/mol.groupname,  symmetry broken might
#   occur due to this approximation,  e.g.  with the default ah_start_tol,
#   C2 (16o, 8e) under D2h symmetry might break the degeneracy between
#   pi_x, pi_y orbitals since pi_x, pi_y belong to different irreps.  It can
#   be fixed by increasing the accuracy of AH solver, e.g.
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-10
# * Classic AH can be simulated by setting eg
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
# ah_grad_trust_region allow gradients being increased in AH optimization
    ah_start_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)

    internal_rotation = getattr(__config__, 'mcscf_mc1step_CASSCF_internal_rotation', False)
    ci_response_space = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_response_space', 4)
    ci_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_grad_trust_region', 3.0)
    with_dep4 = getattr(__config__, 'mcscf_mc1step_CASSCF_with_dep4', False)
    chk_ci = getattr(__config__, 'mcscf_mc1step_CASSCF_chk_ci', False)
    kf_interval = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

    ao2mo_level = getattr(__config__, 'mcscf_mc1step_CASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'mcscf_mc1step_CASSCF_natorb', False)
    canonicalization = getattr(__config__, 'mcscf_mc1step_CASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'mcscf_mc1step_CASSCF_sorting_mo_energy', False)
    scale_restoration = getattr(__config__, 'mcscf_mc1step_CASSCF_scale_restoration', 0.5)
    small_rot_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_small_rot_tol', 0.01)
    extrasym = None
    callback = None

    _keys = {
        'max_stepsize', 'max_cycle_macro', 'max_cycle_micro', 'conv_tol',
        'conv_tol_grad', 'ah_level_shift', 'ah_conv_tol', 'ah_max_cycle',
        'ah_lindep', 'ah_start_tol', 'ah_start_cycle', 'ah_grad_trust_region',
        'internal_rotation', 'ci_response_space', 'ci_grad_trust_region',
        'with_dep4', 'chk_ci', 'kf_interval', 'kf_trust_region',
        'fcisolver_max_cycle', 'fcisolver_conv_tol', 'natorb',
        'canonicalization', 'sorting_mo_energy', 'scale_restoration',
        'small_rot_tol', 'extrasym', 'callback',
        'frozen', 'chkfile', 'fcisolver', 'e_tot', 'e_cas', 'ci', 'mo_coeff',
        'mo_energy', 'converged',
    }

    def __init__(self, mf_or_mol, ncas=0, nelecas=0, ncore=None, frozen=None):
        casci.CASBase.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_mc1step_CASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_mc1step_CASSCF_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('chkfile', 'callback'))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        if self.extrasym is not None:
            log.info('Extra symmetry labels:\n%s', str(self.extrasym))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('max_cycle_micro = %d', self.max_cycle_micro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('orbital rotation max_stepsize = %g', self.max_stepsize)
        log.info('orbital rotation threshold for CI restart = %g', self.small_rot_tol)
        log.info('augmented hessian ah_max_cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian ah_conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian ah_linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian ah_level shift = %g', self.ah_level_shift)
        log.info('augmented hessian ah_start_tol = %g', self.ah_start_tol)
        log.info('augmented hessian ah_start_cycle = %d', self.ah_start_cycle)
        log.info('augmented hessian ah_grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_trust_region = %g', self.kf_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('ci_response_space = %d', self.ci_response_space)
        log.info('ci_grad_trust_region = %d', self.ci_grad_trust_region)
        log.info('with_dep4 %d', self.with_dep4)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('ao2mo_level = %d', self.ao2mo_level)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(self.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASSCF are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found at SCF level but not applied to the CASSCF object.
The SCF solvent model will not be applied to the current CASSCF calculation.
To enable the solvent model for CASSCF, the following code needs to be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital energies (diagonal elements of general Fock matrix).

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        if ci0 is None:
            ci0 = self.ci

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)
        e_tot, e_cas, fcivec = casci.kernel(fcasci, mo_coeff, ci0, log,
                                            envs=envs)
        if not isinstance(e_cas, (float, numpy.number)):
            raise RuntimeError('Multiple roots are detected in fcisolver.  '
                               'CASSCF does not know which state to optimize.\n'
                               'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        elif numpy.ndim(e_cas) != 0:
            # This is a workaround for external CI solver compatibility.
            e_cas = e_cas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %#.15g', e_cas)

            if getattr(self.fcisolver, 'spin_square', None):
                try:
                    ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                except NotImplementedError:
                    ss = None
            else:
                ss = None

            if 'imicro' in envs:  # Within CASSCF iteration
                if ss is None:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'])
                else:
                    log.info('macro iter %3d (%3d JK  %3d micro), '
                             'CASSCF E = %#.15g  dE = % .8e  S^2 = %.7f',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'], ss[0])
                if 'norm_gci' in envs and envs['norm_gci'] is not None:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'],
                             envs['norm_gci'], envs['norm_ddm'], envs['max_offdiag_u'])
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                             envs['norm_gorb0'], envs['norm_ddm'], envs['max_offdiag_u'])
            else:  # Initialization step
                if ss is None:
                    log.info('CASCI E = %#.15g', e_tot)
                else:
                    log.info('CASCI E = %#.15g  S^2 = %.7f', e_tot, ss[0])
        return e_tot, e_cas, fcivec

    as_scanner = as_scanner


    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        if self.internal_rotation:
            mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        if self.extrasym is not None:
            extrasym = numpy.asarray(self.extrasym)
            # Allow rotation only if extra symmetry labels are the same
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask * extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return numpy.dot(u0, expmat(dr))

    gen_g_hop = gen_g_hop
    rotate_orb_cc = rotate_orb_cc

    def update_ao2mo(self, mo):
        raise DeprecationWarning('update_ao2mo was obsoleted since pyscf v1.0.  '
                                 'Use .ao2mo method instead')

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
#        nmo = mo.shape[1]
#        ncore = self.ncore
#        ncas = self.ncas
#        nocc = ncore + ncas
#        eri = pyscf.ao2mo.incore.full(self._scf._eri, mo)
#        eri = pyscf.ao2mo.restore(1, eri, nmo)
#        eris = lambda:None
#        eris.j_cp = numpy.einsum('iipp->ip', eri[:ncore,:ncore,:,:])
#        eris.k_cp = numpy.einsum('ippi->ip', eri[:ncore,:,:,:ncore])
#        eris.vhf_c =(numpy.einsum('iipq->pq', eri[:ncore,:ncore,:,:])*2
#                    -numpy.einsum('ipqi->pq', eri[:ncore,:,:,:ncore]))
#        eris.ppaa = numpy.asarray(eri[:,:,ncore:nocc,ncore:nocc], order='C')
#        eris.papa = numpy.asarray(eri[:,ncore:nocc,:,ncore:nocc], order='C')
#        return eris

        return mc_ao2mo._ERIS(self, mo_coeff, method='incore',
                              level=self.ao2mo_level)

    get_h2eff = CASCI.get_h2eff

    def update_jk_in_ah(self, mo, r, casdm1, eris):
        # J3 = eri_popc * pc + eri_cppo * cp
        # K3 = eri_ppco * pc + eri_pcpo * cp
        # J4 = eri_pcpa * pa + eri_appc * ap
        # K4 = eri_ppac * pa + eri_papc * ap
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        dm3 = reduce(numpy.dot, (mo[:,:ncore], r[:ncore,ncore:], mo[:,ncore:].T))
        dm3 = dm3 + dm3.T
        dm4 = reduce(numpy.dot, (mo[:,ncore:nocc], casdm1, r[ncore:nocc], mo.T))
        dm4 = dm4 + dm4.T
        vj, vk = self.get_jk(self.mol, (dm3,dm3*2+dm4))
        va = reduce(numpy.dot, (casdm1, mo[:,ncore:nocc].T, vj[0]*2-vk[0], mo))
        vc = reduce(numpy.dot, (mo[:,:ncore].T, vj[1]*2-vk[1], mo[:,ncore:]))
        return va, vc

# hessian_co exactly expands up to first order of H
# update_casdm exand to approx 2nd order of H
    def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
        nmo = mo.shape[1]
        rmat = u - numpy.eye(nmo)

        #g = hessian_co(self, mo, rmat, fcivec, e_cas, eris)
        ### hessian_co part start ###
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        uc = u[:,:ncore]
        ua = u[:,ncore:nocc].copy()
        ra = rmat[:,ncore:nocc].copy()
        h1e_mo = reduce(numpy.dot, (mo.T, self.get_hcore(), mo))
        ddm = numpy.dot(uc, uc.T) * 2
        ddm[numpy.diag_indices(ncore)] -= 2
        if self.with_dep4:
            mo1 = numpy.dot(mo, u)
            mo1_cas = mo1[:,ncore:nocc]
            dm_core = numpy.dot(mo1[:,:ncore], mo1[:,:ncore].T) * 2
            vj, vk = self._scf.get_jk(self.mol, dm_core)
            h1 =(reduce(numpy.dot, (ua.T, h1e_mo, ua)) +
                 reduce(numpy.dot, (mo1_cas.T, vj-vk*.5, mo1_cas)))
            eris._paaa = self._exact_paaa(mo, u)
            h2 = eris._paaa[ncore:nocc]
            vj = vk = None
        else:
            p1aa = numpy.empty((nmo,ncas,ncas**2))
            paa1 = numpy.empty((nmo,ncas**2,ncas))
            jk = reduce(numpy.dot, (ua.T, eris.vhf_c, ua))
            for i in range(nmo):
                jbuf = eris.ppaa[i]
                kbuf = eris.papa[i]
                jk += (numpy.einsum('quv,q->uv', jbuf, ddm[i]) -
                       numpy.einsum('uqv,q->uv', kbuf, ddm[i]) * .5)
                p1aa[i] = lib.dot(ua.T, jbuf.reshape(nmo,-1))
                paa1[i] = lib.dot(kbuf.transpose(0,2,1).reshape(-1,nmo), ra)
            h1 = reduce(numpy.dot, (ua.T, h1e_mo, ua)) + jk
            aa11 = lib.dot(ua.T, p1aa.reshape(nmo,-1)).reshape((ncas,)*4)
            aaaa = eris.ppaa[ncore:nocc,ncore:nocc,:,:]
            aa11 = aa11 + aa11.transpose(2,3,0,1) - aaaa

            a11a = numpy.dot(ra.T, paa1.reshape(nmo,-1)).reshape((ncas,)*4)
            a11a = a11a + a11a.transpose(1,0,2,3)
            a11a = a11a + a11a.transpose(0,1,3,2)
            h2 = aa11 + a11a
            jbuf = kbuf = p1aa = paa1 = aaaa = aa11 = a11a = None

        # pure core response
        # response of (1/2 dm * vhf * dm) ~ ddm*vhf
# Should I consider core response as a part of CI gradients?
        ecore =(numpy.einsum('pq,pq->', h1e_mo, ddm) +
                numpy.einsum('pq,pq->', eris.vhf_c, ddm))
        ### hessian_co part end ###

        ci1, g = self.solve_approx_ci(h1, h2, fcivec, ecore, e_cas, envs)
        if g is not None:  # So state average CI, DMRG etc will not be applied
            ovlp = numpy.dot(fcivec.ravel(), ci1.ravel())
            norm_g = numpy.linalg.norm(g)
            if 1-abs(ovlp) > norm_g * self.ci_grad_trust_region:
                logger.debug(self, '<ci1|ci0>=%5.3g |g|=%5.3g, ci1 out of trust region',
                             ovlp, norm_g)
                ci1 = fcivec.ravel() + g
                ci1 *= 1/numpy.linalg.norm(ci1)
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, ncas, nelecas)

        return casdm1, casdm2, g, ci1

    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs):
        ''' Solve CI eigenvalue/response problem approximately
        '''
        ncas = self.ncas
        nelecas = self.nelecas
        if 'norm_gorb' in envs:
            tol = max(self.conv_tol, envs['norm_gorb']**2*.1)
        else:
            tol = None
        if getattr(self.fcisolver, 'approx_kernel', None):
            fn = self.fcisolver.approx_kernel
            e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory)
            return ci1, None
        elif not (getattr(self.fcisolver, 'contract_2e', None) and
                  getattr(self.fcisolver, 'absorb_h1e', None)):
            fn = self.fcisolver.kernel
            e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory,
                        max_cycle=self.ci_response_space)
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)

        def contract_2e(c):
            hc = self.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
            return hc.ravel()

        hc = contract_2e(ci0)
        g = hc - (e_cas-ecore) * ci0.ravel()

        if self.ci_response_space > 7 or ci0.size <= self.fcisolver.pspace_size:
            logger.debug(self, 'CI step by full response')
            # full response
            max_memory = max(400, self.max_memory-lib.current_memory()[0])
            e, ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ecore=ecore,
                                           ci0=ci0, tol=tol, max_memory=max_memory)
        else:
            nd = min(self.ci_response_space, ci0.size)
            xs = [ci0.ravel()]
            ax = [hc]
            heff = numpy.empty((nd,nd))
            seff = numpy.empty((nd,nd))
            heff[0,0] = numpy.dot(xs[0], ax[0])
            seff[0,0] = 1
            tol_residual = self.fcisolver.conv_tol ** .5
            for i in range(1, nd):
                dx = ax[i-1] - xs[i-1] * e_cas
                if numpy.linalg.norm(dx) < tol_residual:
                    break
                xs.append(dx)
                ax.append(contract_2e(xs[i]))
                for j in range(i+1):
                    heff[i,j] = heff[j,i] = numpy.dot(xs[i], ax[j])
                    seff[i,j] = seff[j,i] = numpy.dot(xs[i], xs[j])
            nd = len(xs)
            e, v, seig = lib.safe_eigh(heff[:nd,:nd], seff[:nd,:nd])
            ci1 = xs[0] * v[0,0]
            for i in range(1, nd):
                ci1 += xs[i] * v[i,0]
        return ci1, g

    def get_grad(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        '''Orbital gradients'''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if eris is None: eris = self.ao2mo(mo_coeff)
        if casdm1_casdm2 is None:
            e_tot, e_cas, civec = self.casci(mo_coeff, self.ci, eris)
            casdm1, casdm2 = self.fcisolver.make_rdm12(civec, self.ncas, self.nelecas)
        else:
            casdm1, casdm2 = casdm1_casdm2
        return self.gen_g_hop(mo_coeff, 1, casdm1, casdm2, eris)[0]

    def _exact_paaa(self, mo, u, out=None):
        nmo = mo.shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        mo1 = numpy.dot(mo, u)
        mo1_cas = mo1[:,ncore:nocc]
        mos = (mo1_cas, mo1_cas, mo1, mo1_cas)
        if self._scf._eri is None:
            aapa = ao2mo.general(self.mol, mos)
        else:
            aapa = ao2mo.general(self._scf._eri, mos)
        paaa = numpy.empty((nmo*ncas,ncas*ncas))
        buf = numpy.empty((ncas,ncas,nmo*ncas))
        for ij, (i, j) in enumerate(zip(*numpy.tril_indices(ncas))):
            buf[i,j] = buf[j,i] = aapa[ij]
        paaa = lib.transpose(buf.reshape(ncas*ncas,-1), out=out)
        return paaa.reshape(nmo,ncas,ncas,ncas)

    def dump_chk(self, envs_or_file):
        '''Serialize the MCSCF object and save it to the specified chkfile.

        Args:
            envs_or_file:
                If this argument is a file path, the serialized MCSCF object is
                saved to the file specified by this argument.
                If this attribute is a dict (created by locals()), the necessary
                variables are saved to the file specified by the attribute .chkfile.
        '''
        if isinstance(envs_or_file, str):
            envs = None
            chk_file = envs_or_file
        else:
            envs = envs_or_file
            chk_file = self.chkfile
            if not chk_file:
                return self

        e_tot = mo_coeff = mo_occ = mo_energy = e_cas = civec = casdm1 = None
        ncore = self.ncore
        nocc = ncore + self.ncas

        if envs is not None:
            if self.chk_ci:
                civec = envs.get('fcivec', None)

            e_tot = envs['e_tot']
            e_cas = envs['e_cas']
            casdm1 = envs['casdm1']
            if 'mo' in envs:
                mo_coeff = envs['mo']
            else:
                mo_coeff = envs['mo_coeff']
            mo_occ = numpy.zeros(mo_coeff.shape[1])
            mo_occ[:ncore] = 2
            if self.natorb:
                occ = self._eig(-casdm1, ncore, nocc)[0]
                mo_occ[ncore:nocc] = -occ
            else:
                mo_occ[ncore:nocc] = casdm1.diagonal()
            # Note: mo_energy in active space =/= F_{ii}  (F is general Fock)
            if 'mo_energy' in envs:
                mo_energy = envs['mo_energy']

        chkfile.dump_mcscf(self, chk_file, 'mcscf', e_tot,
                           mo_coeff, ncore, self.ncas, mo_occ,
                           mo_energy, e_cas, civec, casdm1,
                           overwrite_mol=(envs is None))
        return self

    def update_from_chk(self, chkfile=None):
        if chkfile is None: chkfile = self.chkfile
        self.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
        return self
    update = update_from_chk

    def rotate_mo(self, mo, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        mo = numpy.dot(mo, u)
        if log is not None and log.verbose >= logger.DEBUG:
            ncore = self.ncore
            ncas = self.ncas
            nocc = ncore + ncas
            s = reduce(numpy.dot, (mo[:,ncore:nocc].T, self._scf.get_ovlp(),
                                   self.mo_coeff[:,ncore:nocc]))
            log.debug('Active space overlap to initial guess, SVD = %s',
                      numpy.linalg.svd(s)[1])
            log.debug('Active space overlap to last step, SVD = %s',
                      numpy.linalg.svd(u[ncore:nocc,ncore:nocc])[1])
        return mo

    def micro_cycle_scheduler(self, envs):
        if not WITH_MICRO_SCHEDULER:
            return self.max_cycle_micro

        log_norm_ddm = numpy.log(envs['norm_ddm'])
        return max(self.max_cycle_micro, int(self.max_cycle_micro-1-log_norm_ddm))

    max_stepsize_scheduler=max_stepsize_scheduler

    def ah_scheduler(self, envs):
        pass

    @property
    def max_orb_stepsize(self):  # pragma: no cover
        return self.max_stepsize
    @max_orb_stepsize.setter
    def max_orb_stepsize(self, x):  # pragma: no cover
        sys.stderr.write('WARN: Attribute "max_orb_stepsize" was replaced by "max_stepsize"\n')
        self.max_stepsize = x
    @property
    def ci_update_dep(self):  # pragma: no cover
        return self.with_dep4
    @ci_update_dep.setter
    def ci_update_dep(self, x):  # pragma: no cover
        sys.stderr.write('WARN: Attribute .ci_update_dep was replaced by .with_dep4 since PySCF v1.1.\n')
        self.with_dep4 = x == 4
    grad_update_dep = ci_update_dep

    @property
    def max_cycle(self):
        return self.max_cycle_macro
    @max_cycle.setter
    def max_cycle(self, x):
        self.max_cycle_macro = x

    def approx_hessian(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.approx_hessian(self, auxbasis, with_df)

    def nuc_grad_method(self):
        from pyscf.grad import casscf
        return casscf.Gradients(self)

    def _state_average_nuc_grad_method (self, state=None):
        # Hook for addons.state_average. Every child method of CASSCF will
        # probably need to overwrite this.
        from pyscf.grad import sacasscf as sacasscf_grad
        return sacasscf_grad.Gradients (self, state=state)

    def _state_average_nac_method(self):
        from pyscf.nac import sacasscf as sacasscf_nac
        return sacasscf_nac.NonAdiabaticCouplings(self)

    def newton(self):
        from pyscf.mcscf import newton_casscf
        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        mc1 = newton_casscf.CASSCF(self._scf, self.ncas, self.nelecas)
        mc1.__dict__.update(self.__dict__)
        mc1.max_cycle_micro = 10
        # MRH, 04/08/2019: enable state-average CASSCF second-order algorithm
        if isinstance(self, StateAverageMCSCFSolver):
            # FIXME: (QS) Should not need to pass wfnsym for general CASSCF object.
            wfnsym = getattr(self, 'wfnsym', None)
            mc1 = mc1.state_average_(self.weights, wfnsym)
        return mc1

    def reset(self, mol=None):
        casci.CASBase.reset(self, mol=mol)
        self._max_stepsize = None

    to_gpu = lib.to_gpu

scf.hf.RHF.CASSCF = scf.rohf.ROHF.CASSCF = lib.class_as_method(CASSCF)
scf.uhf.UHF.CASSCF = None


# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(CASCI)
    mc.mo_coeff = mo

    if eris is None:
        return mc

    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:nocc]
    core_dm = numpy.dot(mo_core, mo_core.T) * 2
    hcore = casscf.get_hcore()
    energy_core = casscf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace()
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore, mo_cas))
    h1eff += eris.vhf_c[ncore:nocc,ncore:nocc]
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc

def expmat(a):
    return scipy.linalg.expm(a)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import fci
    from pyscf.mcscf import addons

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
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
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
    mc.natorb = 1
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
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
    mc.fcisolver = fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    #mc.analyze()
    print(emc - -75.7155632535814)

    mc.internal_rotation = True
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)
