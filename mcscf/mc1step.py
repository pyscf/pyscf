#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import time
import copy
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.gto
import pyscf.lib.logger as logger
import pyscf.scf
from pyscf.mcscf import casci
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile

# ref. JCP, 82, 5053;  JCP, 73, 2342

def h1e_for_cas(casscf, mo, eris):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    if ncore == 0:
        vhf_c = 0
    else:
        vhf_c = eris.vhf_c[ncore:nocc,ncore:nocc]
    mocc = mo[:,ncore:nocc]
    h1eff = reduce(numpy.dot, (mocc.T, casscf.get_hcore(), mocc)) + vhf_c
    return h1eff

def expmat(a):
    return scipy.linalg.expm(a)

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    # part2, part3
    vhf_a =(numpy.einsum('uvpq,uv->pq', eris.aapp, casdm1)
          - numpy.einsum('upqv,uv->pq', eris.appa, casdm1) * .5)
    vhf_ca = eris.vhf_c + vhf_a

    ################# gradient #################
    #hdm2 = numpy.einsum('tuvw,vwpq->tupq', casdm2, eris.aapp)
    hdm2 = pyscf.lib.dot(casdm2.reshape(ncas*ncas,-1), \
                         eris.aapp.reshape(ncas*ncas,-1)).reshape(ncas,ncas,nmo,nmo)

    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))
    g = numpy.zeros_like(h1e_mo)
    g[:,:ncore] = (h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
    g[:,ncore:nocc] = numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc],casdm1)
    g[:,ncore:nocc] += numpy.einsum('vuuq->qv',hdm2[:,:,ncore:nocc])

    def gorb_update(u, dr):
        if casscf.grad_update_dep == 0: # FEP0/first order R-expansion
            #dr = casscf.pack_uniq_var(u)
            return g_orb + h_op1(dr) + h_opjk(dr)
        else: # DEP1/first order T-expansion
            dt = u - numpy.eye(u.shape[0])
            mo1 = numpy.dot(mo, dt)
            g = numpy.dot(h1e_mo, dt)
            g = h1e_mo + g + g.T
            g[:,nocc:] = 0
            mo_core = mo[:,:ncore]
            mo_cas = mo[:,ncore:nocc]
            dm_core = numpy.dot(mo_core, mo1[:,:ncore].T) * 2
            dm_core = dm_core + dm_core.T
            dm_cas = reduce(numpy.dot, (mo_cas, casdm1, mo1[:,ncore:nocc].T))
            dm_cas = dm_cas + dm_cas.T
            vj, vk = casscf._scf.get_jk(casscf.mol, (dm_core,dm_cas))
            vhfc = numpy.dot(eris.vhf_c, dt)
            vhfc = (vhfc + vhfc.T + eris.vhf_c
                    + reduce(numpy.dot, (mo.T, vj[0]-vk[0]*.5, mo)))
            vhfa = numpy.dot(vhf_a, dt)
            vhfa = (vhfa + vhfa.T + vhf_a
                    + reduce(numpy.dot, (mo.T, vj[1]-vk[1]*.5, mo)))
            g[:,:ncore] += vhfc[:,:ncore]+vhfa[:,:ncore]
            g[:,:ncore] *= 2
            g[:,ncore:nocc] = numpy.dot(g[:,ncore:nocc]+vhfc[:,ncore:nocc], casdm1)

            dt = dt[:,ncore:nocc].copy()
            tmp = numpy.empty((ncas,ncas,nmo*ncas))
            for i in range(ncas):
                tmp[i] = pyscf.lib.dot(dt.T, eris.appa[i].reshape(nmo,-1))
            tmp = tmp + tmp.transpose(1,0,2)
            tmp = numpy.einsum('tuvw,tuqw->qv', casdm2,
                               tmp.reshape(ncas,ncas,nmo,ncas))
            hdm2 = pyscf.lib.dot(casdm2.reshape(ncas*ncas,-1), \
                                 eris.aapp.reshape(ncas*ncas,-1))
            hdm2 = hdm2.reshape(ncas,ncas,nmo,nmo)
            tmp += numpy.einsum('vupq,pu->qv', hdm2, dt)
            tmp += numpy.dot(u.T, numpy.einsum('vuuq->qv',hdm2[:,:,ncore:nocc]))
            g[:,ncore:nocc] += tmp
            return casscf.pack_uniq_var(g-g.T)

    ############## hessian, diagonal ###########
    # part1
    tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3) # (jp|rk) *[E(jq,sk) + E(jq,ks)] => qspr
    tmp = numpy.einsum('uvtw,tpqw->upvq', tmp, eris.appa)
    hdm2 = tmp + hdm2.transpose(0,2,1,3)

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

    # part4
    # -2(pr|sq) + 4(pq|sr) + 4(pq|rs) - 2(ps|rq)
    tmp = 6 * eris.k_cp - 2 * eris.j_cp
    h_diag[:,:ncore] += tmp.T
    h_diag[:ncore,:] += tmp

    # part5 and part6 diag
    # -(qr|kp) E_s^k  p in core, sk in active
    jc_aa = numpy.einsum('uvii->iuv', eris.aapp[:,:,:nocc,:nocc])
    kc_aa = numpy.einsum('uiiv->iuv', eris.appa[:,:nocc,:nocc,:])
    tmp = numpy.einsum('jik,ik->ji', 6*kc_aa-2*jc_aa, casdm1)
    h_diag[:nocc,ncore:nocc] -= tmp
    h_diag[ncore:nocc,:nocc] -= tmp.T

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag
    h_diag[:,ncore:nocc] += v_diag.T

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op1(x):
        x1 = casscf.unpack_uniq_var(x)

        # part7
        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(numpy.dot, (h1e_mo, x1, dm1))
        # part8
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        x2 -= numpy.dot(g.T, x1)
        # part2
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += numpy.dot(x1[:ncore,ncore:], vhf_ca[ncore:]) * 2
        # part3
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x1[ncore:nocc], eris.vhf_c))
        # part1
        x2[ncore:nocc] += numpy.einsum('upvr,vr->up', hdm2, x1[ncore:nocc])

        # from (pr<->qs)
        x2 = x2 - x2.T
        return casscf.pack_uniq_var(x2)

    def h_opjk(x):
        if ncore > 0:
            # part4, part5, part6
            x1 = casscf.unpack_uniq_var(x)
            x2 = numpy.zeros_like(x1)
# Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
#    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
# x2[:,:ncore] += H * x1[:,:ncore] => (becuase x1=-x1.T) =>
# x2[:,:ncore] += -H' * x1[:ncore] => (becuase x2-x2.T) =>
# x2[:ncore] += H' * x1[:ncore]
            va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc
            x2 = x2 - x2.T
            return casscf.pack_uniq_var(x2)
        else:
            return 0

    return g_orb, gorb_update, h_op1, h_opjk, h_diag


def rotate_orb_cc(casscf, mo, casdm1, casdm2, eris, x0_guess=None, verbose=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(casscf.stdout, casscf.verbose)

    t2m = (time.clock(), time.time())
    g_orb, gorb_update, h_op1, h_opjk, h_diag = \
            casscf.gen_g_hop(mo, casdm1, casdm2, eris)
    norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|=%4.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t2m)

    def precond(x, e):
        hdiagd = h_diag-(e-casscf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        norm_x = numpy.linalg.norm(x)
        x *= 1/norm_x
        #if norm_x < 1e-2:
        #    x *= 1e-2/norm_x
        return x

# Dynamically increase the number of micro cycles when approach convergence?
    if norm_gorb < 0.01:
        max_cycle = casscf.max_cycle_micro_inner-int(numpy.log10(norm_gorb+1e-9))
    else:
        max_cycle = casscf.max_cycle_micro_inner

    xsinit = []
    axinit = []
    xcollect = []
    jkcollect = []
    x0 = 0
    u = 1
    jkcount = 0
    norm_dr = 0
    if x0_guess is None:
        x0_guess = g_orb
    ah_conv_tol = min(norm_gorb**2, casscf.ah_conv_tol)
    ah_start_tol = (numpy.log(norm_gorb+casscf.conv_tol_grad) -
                    numpy.log(min(norm_gorb,casscf.conv_tol_grad))) * 1.5 * norm_gorb
    ah_start_tol = max(min(ah_start_tol, casscf.ah_start_tol), ah_conv_tol)
    while True:
        # increase the AH accuracy when approach convergence
        ah_start_tol = max(norm_gorb**2, ah_start_tol)
        #ah_start_cycle = max(casscf.ah_start_cycle, int(-numpy.log10(norm_gorb)))
        ah_start_cycle = casscf.ah_start_cycle
        log.debug('Set ah_start_tol %g, ah_start_cycle %d, max_cycle %d',
                  ah_start_tol, ah_start_cycle, max_cycle)
        g_orb0 = g_orb
        norm_gprev = norm_gorb
        imic = 0
        dx = 0

        g_op = lambda: g_orb
        def h_op(x):
            jk = h_opjk(x)
            if len(xcollect) < casscf.ah_guess_space:
                xcollect.append(x)
                jkcollect.append(jk)
            return h_op1(x) + jk
# Divide the hessian into two parts, approx the JK part:
# In the same macro iteration, the change in JK should be small.  The reason
# is that JK is associated with the core DM1 and active space DM1.  The core
# DM is not changed because eris are not changed, only the active space DM1
# are slightly changed due to the update_casdm function
        if (casscf.ah_guess_space and len(xcollect) > 0 and
            norm_gorb > casscf.conv_tol_grad*5e1):
            xsinit = [x for x in xcollect]
            axinit = [h_op1(x)+jkcollect[i] for i,x in enumerate(xcollect)]
            xcollect = []
            jkcollect = []

        for ah_end, ihop, w, dxi, hdxi, residual, seig \
                in davidson_cc(h_op, g_op, precond, x0_guess,
                               xs=xsinit, ax=axinit, verbose=log,
                               tol=ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                               lindep=casscf.ah_lindep):
            norm_residual = numpy.linalg.norm(residual)
            if (ah_end or ihop == casscf.ah_max_cycle or # make sure to use the last step
                ((norm_residual < ah_start_tol) and (ihop >= ah_start_cycle)) or
                (seig < casscf.ah_lindep)):
                imic += 1
                dxmax = numpy.max(abs(dxi))
                if dxmax > casscf.max_stepsize:
                    scale = casscf.max_stepsize / dxmax
                    log.debug1('... scale rotation size %g', scale)
                    dxi *= scale
                    hdxi *= scale
                else:
# Gradually decrease start_tol/conv_tol, so the next step is more accurate
                    ah_start_tol *= casscf.ah_decay_rate
                    #ah_start_tol *= imic/(1/casscf.ah_decay_rate-1+imic)
                    log.debug('Set ah_start_tol %g', ah_start_tol)
                dx = dx + dxi
                g_orb1 = g_orb + hdxi
                #g_orb1 = g_orb + h_op1(dxi) + h_opjk(dxi)
                #jkcount += 1

                norm_gorb = numpy.linalg.norm(g_orb1)
                norm_dxi = numpy.linalg.norm(dxi)
                norm_dr = numpy.linalg.norm(x0+dx)
                log.debug('    imic %d(%d)  |g[o]|= %4.3g  |dxi|= %4.3g  '
                          'max(|x|)= %4.3g  |dr|= %4.3g  eig= %4.3g  |v|= %4.3g  seig= %4.3g',
                          imic, ihop, norm_gorb, norm_dxi,
                          dxmax, norm_dr, w, norm_residual, seig)

                if norm_gorb > norm_gprev * casscf.ah_grad_trust_region:
# Do we need force the gradients decaying?
# If in the concave region, how to avoid steping backward (along the negative hessian)?
                    dx -= dxi
                    log.debug('    norm_gorb > nrom_gorb_prev')
                    if numpy.linalg.norm(dx) > 1e-14:
                        break
                else:
                    norm_gprev = norm_gorb
                    g_orb = g_orb1
                    u = casscf.update_rotate_matrix(dxi, u)

                if (imic >= max_cycle or norm_gorb < casscf.conv_tol_grad*.8):
                    break

# It's better to exclude the pseudo-linear-dependent trial vectors before the
# next cycle of orbital rotation since these vectors might break
# scipy.linalg.eigh or stop davidson_cc early before updating the solutions
            if seig < casscf.ah_lindep*1e2 and xcollect:
                xcollect.pop(-1)
                jkcollect.pop(-1)
                log.debug1('... pop xcollect, seig = %g, len(xcollect) = %d',
                           seig, len(xcollect))

        if numpy.linalg.norm(dx) > 1e-14:
            x0 = x0 + dx
        else:
# Occasionally, all trial rotation goes to the case "norm_gorb > norm_gprev".
# It leads to the orbital rotation being stuck at x0=0
            dxi *= .1
            dx = dxi
            x0 = x0 + dxi
            u = casscf.update_rotate_matrix(dxi, u)
            g_orb = g_orb + hdxi * .1
            #g_orb = g_orb + h_op1(dxi) + h_opjk(dxi)
            #jkcount += 1
            norm_gorb = numpy.linalg.norm(g_orb)
            log.debug('orbital rotation step not found, try to guess |g[o]|= %4.3g  |dx|= %4.3g',
                      norm_gorb, numpy.linalg.norm(dxi))

        jkcount += ihop
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)
        casdm1, casdm2 = (yield u, g_orb0, jkcount)

        g_orb = gorb_update = h_op1 = h_opjk = h_diag = None
        g_orb, gorb_update, h_op1, h_opjk, h_diag = \
                casscf.gen_g_hop(mo, casdm1, casdm2, eris)
        g_orb = gorb_update(u, x0)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.debug('    |g|=%4.3g', norm_gorb)
        x0_guess = dx
        jkcount += 1


def davidson_cc(h_op, g_op, precond, x0, tol=1e-10, xs=[], ax=[],
                max_cycle=30, lindep=1e-14, verbose=logger.WARN):

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = numpy.sqrt(tol)
    # the first trial vector is (1,0,0,...), which is not included in xs
    xs = [x for x in xs]
    ax = [x for x in ax]
    nx = len(xs)
    if nx == 0:
        xs.append(x0)
        ax.append(h_op(x0))
        nx = 1

    heff = numpy.zeros((max_cycle+nx+1,max_cycle+nx+1))
    ovlp = numpy.eye(max_cycle+nx+1)
    w_t = 0
    for istep in range(min(max_cycle,x0.size)):
        g = g_op()
        nx = len(xs)
        for i in range(nx):
            heff[i+1,0] = heff[0,i+1] = numpy.dot(xs[i], g)
            heff[nx,i+1] = heff[i+1,nx] = numpy.dot(xs[nx-1], ax[i])
            ovlp[nx,i+1] = ovlp[i+1,nx] = numpy.dot(xs[nx-1], xs[i])
        nvec = nx + 1
        wlast = w_t
        xtrial, w_t, v_t, index, seig = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs,
                              lindep, log)
        s0 = seig[0]
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - w_t * v_t[0]*xtrial
        norm_dx = numpy.linalg.norm(dx)
        log.debug1('... AH step %d  index= %d  |dx|= %.5g  eig= %.5g  v[0]= %.5g  lindep= %.5g', \
                   istep+1, index, norm_dx, w_t, v_t[0], s0)
        hx *= 1/v_t[0] # == h_op(xtrial)
        if abs(w_t-wlast) < tol and norm_dx < toloose:
            yield True, istep+1, w_t, xtrial, hx, dx, s0
            break
        else:
            yield False, istep+1, w_t, xtrial, hx, dx, s0
        x0 = precond(dx, w_t)
        xs.append(x0)
        ax.append(h_op(x0))


def _regular_step(heff, ovlp, xs, lindep, log):
    w, v, seig = pyscf.lib.safe_eigh(heff, ovlp, lindep)
    log.debug3('AH eigs %s', str(w))

    for index, x in enumerate(abs(v[0])):
        if x > .1:
            break

    if index > 0 and w[index] < -1e-5:
        log.debug1('AH might follow negative hessians %s', str(w[:index]))

    if abs(v[0,index]) < 1e-4:
        raise RuntimeError('aug_hess diverge')
    else:
        w_t = w[index]
        xtrial = _dgemv(v[1:,index]/v[0,index], xs)
        return xtrial, w_t, v[:,index], index, seig

def _dgemv(v, m):
    vm = v[0] * m[0]
    for i,vi in enumerate(v[1:]):
        vm += vi * m[i+1]
    return vm


def kernel(casscf, mo_coeff, tol=1e-7, macro=50, micro=3,
           ci0=None, callback=None, verbose=None,
           dump_chk=True, dump_chk_ci=False):
    '''CASSCF solver
    '''
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 1-step CASSCF')

    mo = mo_coeff
    nmo = mo.shape[1]
    ncas = casscf.ncas
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.ao2mo(mo)
    e_tot, e_ci, fcivec = casscf.casci(mo, ci0, eris)
    log.info('CASCI E = %.15g', e_tot)
    if ncas == nmo:
        log.debug('CASSCF canonicalization')
        mo, fcivec = casscf.canonicalize(mo, fcivec, eris,
                                         cas_natorb=casscf.natorb, verbose=log)
        return True, e_tot, e_ci, fcivec, mo
    conv = False
    toloose = casscf.conv_tol_grad
    totmicro = totinner = 0
    imicro = 0
    norm_gorb = norm_gci = -1
    elast = e_tot
    de = 1e9

    if casscf.diis:
        if isinstance(casscf.diis, pyscf.lib.diis.DIIS):
            adiis = casscf.diis
        else:
            adiis = pyscf.lib.diis.DIIS(casscf)
    dodiis = False
    r0 = None

    t2m = t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    for imacro in range(macro):
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
        t3m = log.timer('CAS DM', *t2m)
        casdm1_old = casdm1

        micro_iter = casscf.rotate_orb_cc(mo, casdm1, casdm2, eris, r0, log)
        if casscf.dynamic_micro_step:
            max_micro = max(micro, int(0-numpy.log10(de+1e-9)))
        else:
            max_micro = micro
        for imicro in range(max_micro):
            if imicro == 0:
                u, g_orb, njk = micro_iter.next()
                norm_gorb0 = norm_gorb = numpy.linalg.norm(g_orb)
            else:
                u, g_orb, njk = micro_iter.send((casdm1,casdm2))
                norm_gorb = numpy.linalg.norm(g_orb)
            norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
            t3m = log.timer('orbital rotation', *t3m)
            casdm1, casdm2, gci = casscf.update_casdm(mo, u, fcivec, e_ci, eris)
            dodiis |= (casscf.diis and imacro > 1 and e_tot - elast > -1e-4)
            if dodiis:
                log.debug('DIIS for casdm1 and casdm2')
                dm12 = numpy.hstack((casdm1.ravel(), casdm2.ravel()))
                dm12 = adiis.update(dm12, xerr=g_orb)
                casdm1 = dm12[:ncas*ncas].reshape(ncas,ncas)
                casdm2 = dm12[ncas*ncas:].reshape((ncas,)*4)

            if isinstance(gci, numpy.ndarray):
                norm_gci = numpy.linalg.norm(gci)
            else:
                norm_gci = -1
            norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
            t3m = log.timer('update CAS DM', *t3m)
            log.debug('micro %d  |u-1|= %4.3g  |g[o]|= %4.3g  ' \
                      '|g[c]|= %4.3g  |ddm|= %4.3g',
                      imicro+1, norm_t, norm_gorb, norm_gci, norm_ddm)

            if callable(callback):
                callback(locals())

            t3m = log.timer('micro iter %d'%(imicro+1), *t3m)
            if (norm_t < toloose or
                (norm_gorb < toloose and norm_ddm < toloose)):
                break

        micro_iter.close()
        micro_iter = None

        totmicro += imicro + 1
        totinner += njk

        r0 = casscf.pack_uniq_var(u)
        mo = numpy.dot(mo, u)

        eris = None
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        elast = e_tot
        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris)
        log.info('macro iter %d (%d JK  %d micro), CASSCF E = %.15g  dE = %.8g',
                 imacro, njk, imicro+1, e_tot, e_tot-elast)
        log.info('               |grad[o]|= %4.3g  |grad[c]|= %4.3g  |ddm|= %4.3g',
                 norm_gorb0, norm_gci, norm_ddm)
        log.debug('CAS space CI energy = %.15g', e_ci)
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de = abs(e_tot - elast)
        if (de < tol
            and (norm_gorb0 < toloose and norm_ddm < toloose)):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if conv: break

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)

    log.debug('CASSCF canonicalization')
    mo, fcivec = casscf.canonicalize(mo, fcivec, eris,
                                     cas_natorb=casscf.natorb, verbose=log)
    if dump_chk:
        casscf.dump_chk(locals())

    log.note('1-step CASSCF, energy = %.15g', e_tot)
    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_ci, fcivec, mo

def get_fock(mc, mo_coeff=None, ci=None, eris=None, verbose=None):
    return casci.get_fock(mc, mo_coeff, ci, eris, verbose)

def cas_natorb(mc, mo_coeff=None, ci=None, eris=None, sort=False, verbose=None):
    return casci.cas_natorb(mc, mo_coeff, ci, eris, sort, verbose)

def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, verbose=logger.NOTE):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mc.stdout, mc.verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if eris is None: eris = mc.ao2mo(mo_coeff)
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_coeff.shape[1]
    fock = mc.get_fock(mo_coeff, ci, eris)
    if cas_natorb:
        mo_coeff1, ci, occ = mc.cas_natorb(mo_coeff, ci, eris, sort=sort,
                                           verbose=verbose)
    else:
# Keep the active space unchanged by default.  The rotation in active space
# may cause problem for external CI solver eg DMRG.
        mo_coeff1 = numpy.empty_like(mo_coeff)
        mo_coeff1[:,ncore:nocc] = mo_coeff[:,ncore:nocc]
    if ncore > 0:
        # note the last two args of ._eig for mc1step_symm
        w, c1 = mc._eig(fock[:ncore,:ncore], 0, ncore)
        if sort:
            idx = numpy.argsort(w)
            w = w[idx]
            c1 = c1[:,idx]
            if hasattr(mc, 'orbsym'): # for mc1step_symm
                mc.orbsym[:ncore] = mc.orbsym[:ncore][idx]
        mo_coeff1[:,:ncore] = numpy.dot(mo_coeff[:,:ncore], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(ncore):
                log.debug('i = %d  <i|F|i> = %12.8f', i+1, w[i])
    if nmo-nocc > 0:
        w, c1 = mc._eig(fock[nocc:,nocc:], nocc, nmo)
        if sort:
            idx = numpy.argsort(w)
            w = w[idx]
            c1 = c1[:,idx]
            if hasattr(mc, 'orbsym'): # for mc1step_symm
                mc.orbsym[nocc:] = mc.orbsym[nocc:][idx]
        mo_coeff1[:,nocc:] = numpy.dot(mo_coeff[:,nocc:], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(nmo-nocc):
                log.debug('i = %d  <i|F|i> = %12.8f', nocc+i+1, w[i])
# still return ci coefficients, in case the canonicalization funciton changed
# cas orbitals, the ci coefficients should also be updated.
    return mo_coeff1, ci


# To extend CASSCF for certain CAS space solver, it can be done by assign an
# object or a module to CASSCF.fcisolver.  The fcisolver object or module
# should at least have three member functions "kernel" (wfn for given
# hamiltonain), "make_rdm12" (1- and 2-pdm), "absorb_h1e" (effective
# 2e-hamiltonain) in 1-step CASSCF solver, and two member functions "kernel"
# and "make_rdm12" in 2-step CASSCF solver
class CASSCF(casci.CASCI):
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
            Max number of micro iterations in each macro iteration.  Depending on
            systems, increasing this value might reduce the total macro
            iterations.  Generally, 2 - 5 steps should be enough.  Default is 3.
        max_cycle_micro_inner : int
            For the augmented hessian solver, max number of orbital rotation
            steps (controled by max_stepsize).  Value between 2 - 8 is preferred.
            Default is 4.
            Note the 1-step CASSCF algorithm prefers many small steps.  Though might
            increasing the total number of iterations, the small steps can effectively
            prevent oscillating in the total energy.  If the (macro iteration) total
            energy does not monotonically decrease, you can try to reduce max_stepsize
            and increase max_cycle_micro_inner.  For simple system, large max_stepsize
            small max_cycle_micro_inner may give the fast convergence, but it's not recommended.
        ah_level_shift : float, for AH solver.
            Level shift for the Davidson diagonalization in AH solver.  Default is 1e-4.
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
        natorb : bool
            Whether to restore the natural orbital in CAS space.  Default is not.
        ci_response_space : int
            subspace size to solve the CI vector response.  Default is 3.
        callback : function
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
    def __init__(self, mf, ncas, nelecas, ncore=None, frozen=[]):
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.frozen = frozen
# the max orbital rotation and CI increment, prefer small step size
        self.max_stepsize = .03
# small max_ci_stepsize is good to converge, since steepest descent is used
#ABORT        self.max_ci_stepsize = .01
#TODO:self.inner_rotation = False # active-active rotation
        self.max_cycle_macro = 50
        self.max_cycle_micro = 3
        self.max_cycle_micro_inner = 4
        self.conv_tol = 1e-7
        self.conv_tol_grad = 1e-4
        # for augmented hessian
        self.ah_level_shift = 1e-5
        self.ah_conv_tol = 1e-12
        self.ah_max_cycle = 30
        self.ah_lindep = 1e-14
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
        self.ah_start_tol = .2
        self.ah_start_cycle = 2
# * Classic AH can be simulated by setting eg
#               max_cycle_micro_inner = 1
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
#               ah_guess_space = 0
# IN EXPERIMENT: ah_grad_trust_region, ah_guess_space, need more tests.
# ah_grad_trust_region allow gradients increase for AH optimization
# ah_guess_space approximate the JK part of hessian from previous steps
# ah_decay_rate gradually improve AH improve by decreasing start_tol/conv_tol
        self.ah_grad_trust_region = 1.5
        self.ah_guess_space = 0
        self.ah_decay_rate = .8
        self.grad_update_dep = 1
        self.ci_update_dep = 2
        self.internal_rotation = False
        self.dynamic_micro_step = False

        self.chkfile = mf.chkfile
        self.ci_response_space = 3
        self.diis = False
        self.natorb = False
        self.callback = None

        self.fcisolver.max_cycle = 50

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.converged = False

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** CASSCF flags ********')
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        if self.frozen:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max. macro cycles = %d', self.max_cycle_macro)
        log.info('max. micro cycles = %d', self.max_cycle_micro)
        log.info('conv_tol = %g, (%g for gradients)',
                 self.conv_tol, self.conv_tol_grad)
        log.info('max_cycle_micro_inner = %d', self.max_cycle_micro_inner)
        log.info('max. orb step = %g', self.max_stepsize)
        #log.info('max. ci step = %g', self.max_ci_stepsize)
        log.info('augmented hessian max_cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian level shift = %d', self.ah_level_shift)
        log.info('augmented hessian start_tol = %g', self.ah_start_tol)
        log.info('augmented hessian start_cycle = %d', self.ah_start_cycle)
        log.info('augmented hessian grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('augmented hessian guess space = %d', self.ah_guess_space)
        log.info('augmented hessian decay rate = %g', self.ah_decay_rate)
        log.info('ci_response_space = %d', self.ci_response_space)
        log.info('diis = %s', self.diis)
        log.info('chkfile = %s', self.chkfile)
        log.info('natorb = %s', self.natorb)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscf.lib.current_memory()[0])
        log.debug('grad_update_dep %d', self.grad_update_dep)
        log.debug('ci_update_dep %d', self.ci_update_dep)
        log.info('dynamic_micro_step %s', self.dynamic_micro_step)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        if hasattr(self, 'max_orb_stepsize'):
            raise AttributeError('"max_orb_stepsize" was replaced by "max_stepsize"')

    def kernel(self, mo_coeff=None, ci0=None, macro=None, micro=None,
               callback=None, _kern=kernel):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None: macro = self.max_cycle_macro
        if micro is None: micro = self.max_cycle_micro
        if callback is None: callback = self.callback

        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.mol.check_sanity(self)
        self.dump_flags()

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, macro=macro, micro=micro,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc1step(self, mo_coeff=None, ci0=None, macro=None, micro=None,
                callback=None):
        return self.kernel(mo_coeff, ci0, macro, micro, callback)

    def mc2step(self, mo_coeff=None, ci0=None, macro=None, micro=1,
                callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, macro, micro, callback,
                           mc2step.kernel)

    def casci(self, mo_coeff, ci0=None, eris=None):
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)
        log = logger.Logger(self.stdout, self.verbose)
        return casci.kernel(fcasci, mo_coeff, ci0=ci0, verbose=log)

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        if self.internal_rotation:
            mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        if frozen:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
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

    def gen_g_hop(self, *args):
        return gen_g_hop(self, *args)

    def rotate_orb_cc(self, mo, casdm1, casdm2, eris, r0, verbose):
        return rotate_orb_cc(self, mo, casdm1, casdm2, eris, r0, verbose)

    def update_ao2mo(self, mo):
        raise RuntimeError('update_ao2mo was obseleted since pyscf v1.0.  Use .ao2mo method instead')

    def ao2mo(self, mo):
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
#        eris.aapp = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
#        eris.appa = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
#        return eris

        if hasattr(self._scf, '_cderi'):
            raise RuntimeError('TODO: density fitting')
        return mc_ao2mo._ERIS(self, mo, 'incore')

    def get_h2eff(self, mo_coeff=None):
        return self.get_h2cas(mo_coeff)
    def get_h2cas(self, mo_coeff=None):
        return casci.CASCI.ao2mo(self, mo_coeff)

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
    def update_casdm(self, mo, u, fcivec, e_ci, eris):
        nmo = mo.shape[1]
        rmat = u - numpy.eye(nmo)

        #g = hessian_co(self, mo, rmat, fcivec, e_ci, eris)
        ### hessian_co part start ###
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        uc = u[:,:ncore]
        ua = u[:,ncore:nocc].copy()
        ra = rmat[:,ncore:nocc].copy()
        rc = rmat[:,:ncore]

        h1e_mo = reduce(numpy.dot, (mo.T, self.get_hcore(), mo))
        ddm = numpy.dot(uc, uc.T) * 2 # ~ dm(1) + dm(2)
        ddm[numpy.diag_indices(ncore)] -= 2
        if self.ci_update_dep == 2: # (0) + (1) + part-of-(2)
            ## missing terms:
            #jk =(numpy.dot(numpy.einsum('upqv,qv->up', eris.appc, rc*2), rc)
            #   - numpy.dot(numpy.einsum('upqv,pv->uq', eris.appc, rc), rc)*.5
            #   - numpy.dot(numpy.einsum('uvpq,pv->uq', eris.acpp, rc), rc)*.5)
            #jk = jk + jk.T
            jk =(reduce(numpy.dot, (ua.T, eris.vhf_c, ua))
               + numpy.einsum('uvpq,pq->uv', eris.aapp, ddm.T)
               - numpy.einsum('upqv,pq->uv', eris.appa, ddm) * .5)
            h1 = reduce(numpy.dot, (ua.T, h1e_mo, ua)) + jk

            paaa = pyscf.lib.dot(eris.appa.transpose(1,0,3,2).reshape(-1,nmo), ra)
            aaa2 = numpy.dot(ra.T, paaa.reshape(nmo,-1)).reshape((ncas,)*4)
            aaa2 = aaa2 + aaa2.transpose(1,0,2,3)
            aaa2 = aaa2 + aaa2.transpose(0,1,3,2)
            aapa = pyscf.lib.dot(eris.aapp.reshape(-1,nmo), ua)
            aaa1 = pyscf.lib.dot(aapa.reshape(-1,nmo,ncas).transpose(0,2,1).reshape(-1,nmo),
                                 ua).reshape((ncas,)*4)
            aaaa = eris.appa[:,ncore:nocc,ncore:nocc,:]
            aaa1 = aaa1 + aaa1.transpose(2,3,0,1) - aaaa
            h2 = aaa1 + aaa2
            paaa = aapa = aaaa = aaa1 = aaa2 = None
        else: # jk(0) + jk(1)
            ddm[:] = 0
            ddm[:,:ncore] = rc * 2
            ddm[:ncore] += rc.T * 2
            jk = numpy.dot(ra.T, eris.vhf_c[:,ncore:nocc])
            jk = jk + jk.T + eris.vhf_c[ncore:nocc,ncore:nocc]
            jk+=(numpy.einsum('uvpq,pq->uv', eris.aapp, ddm.T)
               - numpy.einsum('upqv,pq->uv', eris.appa, ddm) * .5)
            h1 = numpy.dot(ra.T, h1e_mo[:,ncore:nocc])
            h1 = h1 + h1.T + h1e_mo[ncore:nocc,ncore:nocc] + jk

            aapa = pyscf.lib.dot(eris.aapp.reshape(-1,nmo), ra).reshape(ncas,ncas,nmo,ncas)
            aaaa = eris.appa[:,ncore:nocc,ncore:nocc,:]
            aaa1 = aapa[:,:,ncore:nocc,:]
            aaa1 = aaa1 + aaa1.transpose(0,1,3,2)
            aaa1 = aaa1 + aaa1.transpose(2,3,0,1)
            h2 = aaaa + aaa1
            aapa = aaaa = aaa1 = None

# (@)       h2eff = self.fcisolver.absorb_h1e(h1cas, aaaa, ncas, nelecas, .5)
# (@)       g = self.fcisolver.contract_2e(h2eff, fcivec, ncas, nelecas).ravel()

        # pure core response
        # response of (1/2 dm * vhf * dm) ~ ddm*vhf
# Should I consider core response as a part of CI gradients?
        ecore =(numpy.einsum('pq,pq->', h1e_mo, ddm)
              + numpy.einsum('pq,pq->', eris.vhf_c, ddm))
        ### hessian_co part end ###

        ci1, g = self.solve_approx_ci(h1, h2, fcivec, ecore, e_ci)
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, ncas, nelecas)

        return casdm1, casdm2, g

    def solve_approx_ci(self, h1, h2, ci0, ecore, e_ci):
        ''' Solve CI eigenvalue/response problem approximately
        '''
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        if hasattr(self.fcisolver, 'approx_kernel'):
            ci1 = self.fcisolver.approx_kernel(h1, h2, ncas, nelecas, ci0=ci0)[1]
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)
        hc = self.fcisolver.contract_2e(h2eff, ci0, ncas, nelecas).ravel()

#        ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ci0=ci0)[1]   # (!)
# In equation (!), h1 and h2 are approximations to the fully transformed
# hamiltonain wrt rotated MO.  h1 and h2 have only 0th and part of 1st order
# (missing VHF[core DM response]).  Fully solving equation (!) would lead
# to an approximation version of the CI solver in the macro iteration.  This
# can be further approximated by reducing the max_cycle for fcisolver.kernel
# or solving eq (!) in a sub-space.  If the size of subspace is 1, it results
# in the gradient updates.  This is the reason why gradeint updates works very
# well.  Other approximations for eq (!) can be
#
# * Perturbation updating   gci/(e_ci-hci_diag), gci = H^1 ci^0  is the way
#   davidson.dsyev generate new trial vector.  Numerically, perturbation
#   updating is worse than the simple gradeint.
# * Restorsing the davidson.dsyev hessian from previous FCI solver as the
#   approx CI hessian then solving  (H-E*1)dc = g or aug-hessian or H dc = g
#   has not obvious advantage than simple gradeint.

        g = hc - (e_ci-ecore) * ci0.ravel()  # hc-eci*ci0 equals to eqs. (@)
        if self.ci_response_space > 6:
            logger.debug(self, 'CI step by full response')
            # full response
            e, ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ci0=ci0)
        else:
            nd = min(max(self.ci_response_space, 2), ci0.size)
            logger.debug(self, 'CI step by %dD subspace response', nd)
            xs = [ci0.ravel()]
            ax = [hc]
            heff = numpy.empty((nd,nd))
            seff = numpy.empty((nd,nd))
            heff[0,0] = numpy.dot(xs[0], ax[0])
            seff[0,0] = 1
            for i in range(1, nd):
                xs.append(ax[i-1] - xs[i-1] * e_ci)
                ax.append(self.fcisolver.contract_2e(h2eff, xs[i], ncas,
                                                     nelecas).ravel())
                for j in range(i+1):
                    heff[i,j] = heff[j,i] = numpy.dot(xs[i], ax[j])
                    seff[i,j] = seff[j,i] = numpy.dot(xs[i], xs[j])
            e, v = pyscf.lib.safe_eigh(heff, seff)[:2]
            ci1 = 0
            for i in range(nd):
                ci1 += xs[i] * v[i,0]
        return ci1, g

    def get_jk(self, mol, dm, hermi=1):
        return self._scf.get_jk(mol, dm, hermi=1)

    def dump_chk(self, envs):
        if hasattr(self.fcisolver, 'nevpt_intermediate'):
            civec = None
        elif envs['dump_chk_ci']:
            civec = envs['fcivec']
        else:
            civec = None
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        occ, ucas = self._eig(-envs['casdm1'], ncore, nocc)
        mo = envs['mo'].copy()
        mo[:,ncore:nocc] = numpy.dot(mo[:,ncore:nocc], ucas)
        mo_occ = numpy.zeros(mo.shape[1])
        mo_occ[:ncore] = 2
        mo_occ[ncore:nocc] = -occ
        chkfile.dump_mcscf(self.mol, self.chkfile, mo,
                           mcscf_energy=envs['e_tot'], e_cas=envs['e_ci'],
                           ci_vector=civec,
                           iter_macro=(envs['imacro']+1),
                           iter_micro_tot=(envs['totmicro']),
                           converged=envs['conv'], mo_occ=mo_occ)

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     cas_natorb=False, verbose=None):
        return canonicalize(self, mo_coeff, ci, eris, sort, cas_natorb, verbose)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, verbose=None):
        self.mo_coeff, self.ci = canonicalize(self, mo_coeff, ci, eris,
                                              sort, cas_natorb, verbose)
        return self.mo_coeff, self.ci


# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    # vhf for core density matrix
    mo_inv = numpy.dot(mo.T, mc._scf.get_ovlp())
    vhf = reduce(numpy.dot, (mo_inv.T, eris.vhf_c, mo_inv))
    mc.get_veff = lambda *args: vhf

    ncore = casscf.ncore
    nocc = ncore + casscf.ncas
    eri_cas = eris.aapp[:,:,ncore:nocc,ncore:nocc].copy()
    mc.ao2mo = lambda *args: eri_cas
    return mc


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
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
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    emc = kernel(mc, m.mo_coeff, verbose=4)[1]
    print(emc - -15.950852049859-mol.energy_nuc())


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
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    mc.analyze()
    print(emc - -75.7155632535814)

    mc.internal_rotation = True
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)
