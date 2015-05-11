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
    vhf_ca = eris.vhf_c + numpy.einsum('uvpq,uv->pq', eris.aapp, casdm1) \
                        - numpy.einsum('upqv,uv->pq', eris.appa, casdm1) * .5

    ################# gradient #################
    #hdm2 = numpy.einsum('tuvw,vwpq->tupq', casdm2, eris.aapp)
    hdm2 = pyscf.lib.dot(casdm2.reshape(ncas*ncas,-1), \
                         eris.aapp.reshape(ncas*ncas,-1)).reshape(ncas,ncas,nmo,nmo)

    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))
    g = numpy.dot(h1e_mo, dm1)
    g[:,:ncore] += vhf_ca[:,:ncore] * 2
    g[:,ncore:nocc] += numpy.einsum('vuuq->qv',hdm2[:,:,ncore:nocc]) \
            + numpy.dot(eris.vhf_c[:,ncore:nocc],casdm1)

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
    h_diag[:ncore,:ncore] -= tmp[:,:ncore] * 2

    # part5 and part6 diag
    # -(qr|kp) E_s^k  p in core, sk in active
    jc_aa = numpy.einsum('uvii->iuv', eris.aapp[:,:,:ncore,:ncore])
    kc_aa = numpy.einsum('uiiv->iuv', eris.appa[:,:ncore,:ncore,:])
    tmp = numpy.einsum('jik,ik->ji', 6*kc_aa-2*jc_aa, casdm1)
    h_diag[:ncore,ncore:nocc] -= tmp
    h_diag[ncore:nocc,:ncore] -= tmp.T

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag
    h_diag[:,ncore:nocc] += v_diag.T

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op1(x):
        x1 = casscf.unpack_uniq_var(x)
        x_cu = x1[:ncore,ncore:]
        x_av = x1[ncore:nocc,nocc:]
        x_ac = x1[ncore:nocc,:ncore]

        # part7
        x2 = reduce(numpy.dot, (h1e_mo, x1, dm1))
        # part8, the hessian provides
        #x2 -= numpy.dot(g, x1)
        # but it may destroy Hermitian unless g == g.T (the converged g).
        # So symmetrize it with
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        #x2 -= (numpy.dot(g, x1) + numpy.dot(g.T, x1)) * .5
        x2 -= numpy.dot(g+g.T, x1) * .5
        # part2
        x2[:ncore] += numpy.dot(x_cu, vhf_ca[ncore:]) * 2
        # part3
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x_av, eris.vhf_c[nocc:])) \
                        + reduce(numpy.dot, (casdm1, x_ac, eris.vhf_c[:ncore]))

        # part1
        x2[ncore:nocc] += numpy.einsum('upvr,vr->up', hdm2, x1[ncore:nocc])

        x2 = x2 - x2.T
        return casscf.pack_uniq_var(x2)

    def h_opjk(x):
        x1 = casscf.unpack_uniq_var(x)
        x2 = numpy.zeros_like(x1)
        # part4, part5, part6
        if ncore > 0:
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

    return g_orb, h_op1, h_opjk, h_diag


def rotate_orb_cc(casscf, mo, fcasdm1, fcasdm2, eris, verbose=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(casscf.stdout, casscf.verbose)

    nmo = mo.shape[1]

    t2m = (time.clock(), time.time())
    g_orb, h_op1, h_opjk, h_diag = casscf.gen_g_hop(mo, fcasdm1(), fcasdm2(), eris)
    norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|=%4.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t2m)

    def precond(x, e):
        hdiagd = h_diag-(e-casscf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return x/hdiagd

# Dynamically increase the number of micro cycles when approach convergence?
    if norm_gorb < 0.01:
        max_cycle = casscf.max_cycle_micro_inner-int(numpy.log10(norm_gorb+1e-12))
    else:
        max_cycle = casscf.max_cycle_micro_inner

    xcollect = [g_orb]
    jkcollect = [h_opjk(g_orb)]
    x0 = 0
    u = numpy.eye(nmo)
    jkcount = 0
    dx1 = g_orb
    while True:
        norm_gprev = numpy.linalg.norm(g_orb)
        # increase the AH accuracy when approach convergence
        if norm_gprev*.1 < casscf.ah_start_tol:
            ah_start_tol = norm_gprev*.1
            log.debug1('Set AH start tol to %g', ah_start_tol)
        else:
            ah_start_tol = casscf.ah_start_tol
        ah_start_tol = max(ah_start_tol, casscf.ah_conv_tol)
        imic = 0
        wlast = 0
        dx = 0

        g_op = lambda: g_orb
        def h_op(x):
            jk = h_opjk(x)
# exclude possible linear dependent vectors
            if numpy.linalg.norm(x) > casscf.ah_conv_tol:
                xcollect.append(x)
                jkcollect.append(jk)
            return h_op1(x) + jkcollect[-1]
# Divide the hessian into two parts, approx the JK part
        xsinit = [x for x in xcollect]
        axinit = [h_op1(x)+jkcollect[i] for i,x in enumerate(xcollect)]

        for ah_conv, ihop, w, dxi, hdxi, residual, seig \
                in davidson_cc(h_op, g_op, precond, dx1,
                               xs=xsinit, ax=axinit, verbose=log,
                               tol=casscf.ah_conv_tol,
                               max_cycle=casscf.ah_max_cycle,
                               lindep=casscf.ah_lindep):
            if (ah_conv or ihop+1 == casscf.ah_max_cycle or # make sure to use the last step
                ((abs(w-wlast) < ah_start_tol) and
                 (numpy.linalg.norm(residual) < casscf.ah_start_tol) and
                 (ihop >= casscf.ah_start_cycle)) or
                (seig < casscf.ah_lindep)):
                imic += 1
                dx1 = dxi
                dxmax = numpy.max(abs(dx1))
                if dxmax > casscf.max_orb_stepsize:
                    scale = casscf.max_orb_stepsize / dxmax
                    log.debug1('scale rotation size %g', scale)
                    dx1 = dx1 * scale
                    dx = dx + dx1
                    g_orb1 = g_orb + h_op1(dx1) + h_opjk(dx1)
                    jkcount += 1
                else:
                    dx = dx + dx1
                    g_orb1 = g_orb + hdxi  # hdxi not good enough?
                    #g_orb1 = g_orb + h_op1(dx1) + h_opjk(dx1)
                    #jkcount += 1
# Gradually lower the start_tol, so the following steps get more precisely
                    ah_start_tol *= .4

                norm_gorb = numpy.linalg.norm(g_orb1)
                norm_dx1 = numpy.linalg.norm(dx1)
                log.debug('    inner iter %d, |g[o]|=%4.3g, |dx|=%4.3g, max(|x|)=%4.3g, eig=%4.3g',
                           imic, norm_gorb, norm_dx1, dxmax, w)

                if norm_gorb > norm_gprev:
                    dx -= dx1
                    log.debug1('norm_gorb > nrom_gorb_pref')
                    if numpy.linalg.norm(dx) > 1e-14:
                        break
                else:
                    norm_gprev = norm_gorb
                    g_orb = g_orb1
                    dr = casscf.unpack_uniq_var(dx1)
                    u = numpy.dot(u, expmat(dr))

# It's better to exclude the pseudo-linear-dependent trial vectors for the
# next round of orbital rotation since these vectors might break
# scipy.linalg.eigh or stop davidson_cc early before updating the solutions
            if seig < casscf.ah_lindep*1e2:
                xcollect.pop(-1)
                jkcollect.pop(-1)
                break

            if (imic >= max_cycle or norm_gorb < casscf.conv_tol_grad*.5):
                break
            wlast = w

        if numpy.linalg.norm(dx) > 1e-14:
            x0 = x0 + dx
        else:
# Occasionally, all trial rotation goes to the branch "norm_gorb > norm_gprev".
# It leads to the orbital rotation being stuck at x0=0
            dx1 *= .2
            x0 = x0 + dx1
            g_orb = g_orb + h_op1(dx1) + h_opjk(dx1)
            jkcount += 1
            dr = casscf.unpack_uniq_var(dx1)
            u = numpy.dot(u, expmat(dr))
            log.debug('orbital rotation step not found, try to guess |g[o]|=%4.3g, |dx|=%4.3g',
                      numpy.linalg.norm(g_orb), numpy.linalg.norm(dx1))

        jkcount += ihop + 1
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)
        yield u, g_orb, jkcount

        g_orb, h_op1, h_opjk, h_diag = casscf.gen_g_hop(mo, fcasdm1(), fcasdm2(), eris)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.debug('    |g|=%4.3g', norm_gorb)
        g_orb = g_orb + h_op1(x0) + h_opjk(x0)
        jkcount += 1


def davidson_cc(h_op, g_op, precond, x0, tol=1e-7, xs=[], ax=[],
                max_cycle=10, lindep=1e-14, verbose=logger.WARN):

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    # the first trial vector is (1,0,0,...), which is not included in xs
    nx = len(xs)
    if nx == 0:
        xs.append(x0)
        ax.append(h_op(x0))
        nx = 1

    heff = numpy.zeros((max_cycle+nx+1,max_cycle+nx+1))
    ovlp = numpy.eye(max_cycle+nx+1)
    conv = False
    for i,xi in enumerate(xs):
        for j in range(i+1):
            heff[i+1,j+1] = heff[j+1,i+1] = numpy.dot(xi, ax[j])
            ovlp[i+1,j+1] = ovlp[j+1,i+1] = numpy.dot(xi, xs[j])
    for istep in range(min(max_cycle,x0.size)):
        g = g_op()
        nx = len(xs)
        for i in range(nx):
            heff[i+1,0] = heff[0,i+1] = numpy.dot(xs[i], g)
            heff[nx,i+1] = heff[i+1,nx] = numpy.dot(xs[nx-1], ax[i])
            ovlp[nx,i+1] = ovlp[i+1,nx] = numpy.dot(xs[nx-1], xs[i])
        nvec = nx + 1
        xtrial, w_t, v_t, index = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs, log)
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - xtrial * (w_t*v_t[0])
        norm_dx = numpy.linalg.norm(dx)/numpy.sqrt(dx.size)
        s0 = scipy.linalg.eigh(ovlp[:nvec,:nvec])[0][0]
        log.debug1('AH step %d, index=%d, bar|dx|=%.5g, eig=%.5g, v[0]=%.5g, lindep=%.5g', \
                   istep+1, index, norm_dx, w_t, v_t[0], s0)
        if norm_dx < tol or s0 < lindep:
            conv = True
            break
        yield conv, istep, w_t, xtrial, hx, dx, s0
        x0 = precond(dx, w_t)
        xs.append(x0)
        ax.append(h_op(x0))

    if x0.size == 0:
        yield conv, istep, 0, x0, 0, x0, 1
    else:
        yield conv, istep, w_t, xtrial, hx, dx, s0

def _regular_step(heff, ovlp, xs, log):
    w, v = scipy.linalg.eigh(heff, ovlp)
    log.debug2('AH eigs %s', str(w))

    for index, x in enumerate(abs(v[0])):
        if x > .1:
            break

    if w[1] < 0:
        log.debug1('Negative hessian eigenvalue found %s',
                   str(scipy.linalg.eigh(heff[1:,1:], ovlp[1:,1:])[0][:5]))
    if index > 0 and w[0] < -1e-5:
        log.debug('AH might follow negative hessians %s', str(w[:index]))

    if abs(v[0,index]) < 1e-4:
        raise RuntimeError('aug_hess diverge')
    else:
        w_t = w[index]
        xtrial = _dgemv(v[1:,index]/v[0,index], xs)
        return xtrial, w_t, v[:,index], index

def _dgemv(v, m):
    vm = v[0] * m[0]
    for i,vi in enumerate(v[1:]):
        vm += vi * m[i+1]
    return vm

# dc = h_{co} * dr
def hessian_co(casscf, mo, rmat, fcivec, e_ci, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncore + ncas
    mocc = mo[:,:nocc]

    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mocc))
    h1eff = numpy.dot(rmat[:,:nocc].T, h1e_mo)
    h1eff = h1eff + h1eff.T

    apca = eris.appa[:,:,:ncore,:]
    aapc = eris.aapp[:,:,:,:ncore]
    jk = eris.vhf_c[:nocc]
    v1 = numpy.einsum('up,pv->uv', jk[ncore:], rmat[:,ncore:nocc]) \
       + numpy.einsum('uvpi,pi->uv', aapc-apca.transpose(0,3,1,2)*.5,
                      rmat[:,:ncore]) * 2
    h1cas = h1eff[ncore:,ncore:] + (v1 + v1.T)

    aaap = eris.aapp[:,:,ncore:nocc,:]
    aaaa = numpy.einsum('tuvp,pw->tuvw', aaap, rmat[:,ncore:nocc])
    aaaa = aaaa + aaaa.transpose(0,1,3,2)
    aaaa = aaaa + aaaa.transpose(2,3,0,1)
    h2eff = casscf.fcisolver.absorb_h1e(h1cas, aaaa, ncas, nelecas, .5)
    hc = casscf.fcisolver.contract_2e(h2eff, fcivec, ncas, nelecas).ravel()

    # pure core response
    # J from [(i^1i|jj) + (ii^1|jj) + (ii|j^1j) + (ii|jj^1)] has factor 4
    # spin-free (ii| gives another 2*2, which provides a total factor 16
    # times 1/2 for 2e-integrals. So the J^1 part has factor 8, K^1 has 4
    ecore = h1eff[:ncore,:ncore].trace()*2 \
          + numpy.einsum('jp,pj->', jk[:ncore], rmat[:,:ncore])*4
    hc += ecore * fcivec.ravel()
    return hc

# dr = h_{oc} * dc
def hessian_oc(casscf, mo, dci, fcivec, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    nocc = ncore + ncas

    tdm1, tdm2 = casscf.fcisolver.trans_rdm12(dci, fcivec, ncas, nelecas)
    tdm1 = (tdm1 + tdm1.T)
    tdm2 = (tdm2 + tdm2.transpose(1,0,3,2))

    inner1 = numpy.dot(dci.flatten(),fcivec.flatten()) * 2

    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2 * inner1 # p^+ q in core, factor due to <c|0> + <0|c>
    dm1[ncore:nocc,ncore:nocc] = tdm1

    vhf_a = numpy.einsum('uvpq,uv->pq', eris.aapp, tdm1) \
          - numpy.einsum('upqv,uv->pq', eris.appa, tdm1) * .5

    aaap = eris.aapp[:,:,ncore:nocc,:]
    g2dm = numpy.dot(aaap.reshape(-1, nmo).T, tdm2.reshape(-1,ncas))

    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))
    g = numpy.dot(h1e_mo, dm1)
    g[:,:ncore] += vhf_a[:,:ncore] * 2
    g[:,:ncore] += eris.vhf_c[:,:ncore] *(2 * inner1)
    g[:,ncore:nocc] += g2dm + numpy.dot(eris.vhf_c[:,ncore:nocc], tdm1)
    return casscf.pack_uniq_var(g - g.T)


def kernel(casscf, mo_coeff, tol=1e-7, macro=50, micro=3,
           ci0=None, verbose=None,
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
        return True, e_tot, e_ci, fcivec, mo
    conv = False
    toloose = casscf.conv_tol_grad
    totmicro = totinner = 0
    imicro = 0
    norm_gorb = norm_gci = 0
    casdm1_old = 0
    elast = e_tot

    if casscf.diis:
        adiis = pyscf.lib.diis.DIIS(casscf)
    dodiis = False

    t2m = t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    for imacro in range(macro):
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
        t3m = log.timer('CAS DM', *t2m)
        fcasdm1 = lambda: casdm1
        fcasdm2 = lambda: casdm2
        casdm1_old = casdm1
        imicro = 0
        for u, g_orb, njk in casscf.rotate_orb_cc(mo, fcasdm1, fcasdm2, eris,
                                                  verbose=log):
            t3m = log.timer('orbital rotation', *t3m)
            imicro += 1

            casdm1, casdm2, gci = casscf.update_casdm(mo, u, fcivec, e_ci, eris)
            dodiis |= (casscf.diis and imacro > 1 and e_tot - elast > -1e-4)
            if dodiis:
                log.debug('DIIS for casdm1 and casdm2')
                dm12 = numpy.hstack((casdm1.ravel(), casdm2.ravel()))
                dm12 = adiis.update(dm12, xerr=g_orb)
                casdm1 = dm12[:ncas*ncas].reshape(ncas,ncas)
                casdm2 = dm12[ncas*ncas:].reshape((ncas,)*4)

            norm_gorb = numpy.linalg.norm(g_orb)
            if imicro == 1:
                norm_gorb0 = numpy.linalg.norm(g_orb)
            norm_gci = numpy.linalg.norm(gci)
            norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
            casdm1_old = casdm1
            norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
            t3m = log.timer('update CAS DM', *t3m)
            log.debug('micro %d, e_ci = %.12g, |u-1|=%4.3g, |g[o]|=%4.3g, ' \
                      '|g[c]|=%4.3g, |ddm|=%4.3g',
                      imicro, e_ci, norm_t, norm_gorb, norm_gci, norm_ddm)

            t3m = log.timer('micro iter %d'%(imicro+1), *t3m)
            if (norm_t < toloose or norm_gci < toloose or
                (norm_gorb < toloose and norm_ddm < toloose) or
                (imicro >= micro)):
                break

        totmicro += imicro
        totinner += njk

        mo = numpy.dot(mo, u)

        eris = None
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        elast = e_tot
        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris)
        log.info('macro iter %d (%d JK, %d micro), CASSCF E = %.15g, dE = %.8g,',
                 imacro, njk, imicro, e_tot, e_tot-elast)
        log.info('               |grad[o]|=%4.3g, |grad[c]|=%4.3g, |ddm|=%4.3g',
                 norm_gorb0, norm_gci, norm_ddm)
        log.debug('CAS space CI energy = %.15g', e_ci)
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        if abs(e_tot - elast) < tol \
           and (norm_gorb0 < toloose and norm_ddm < toloose):
            conv = True

        if dump_chk:
            casscf.save_mo_coeff(mo, imacro, imicro)
            casscf.dump_chk(mo,
                            mcscf_energy=e_tot, e_cas=e_ci,
                            ci_vector=(fcivec if dump_chk_ci else None),
                            iter_macro=imacro+1,
                            iter_micro_tot=totmicro,
                            converged=(conv if (conv or (imacro+1 >= macro)) else None),
                           )
        if conv: break

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)

    log.debug('CASSCF canonicalization')
    mo, fcivec = casscf.canonicalize(mo, fcivec, eris, log)
    casscf.save_mo_coeff(mo, imacro, imicro)

    log.note('1-step CASSCF, energy = %.15g', e_tot)
    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_ci, fcivec, mo

def get_fock(mc, mo_coeff=None, ci=None, eris=None, verbose=None):
    return casci.get_fock(mc, mo_coeff, ci, eris, verbose=None)

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
        mo_coeff1[:,:ncore] = numpy.dot(mo_coeff[:,:ncore], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(ncore):
                log.debug('i = %d, <i|F|i> = %12.8f', i+1, w[i])
    if nmo-nocc > 0:
        w, c1 = mc._eig(fock[nocc:,nocc:], nocc, nmo)
        if sort:
            idx = numpy.argsort(w)
            w = w[idx]
            c1 = c1[:,idx]
        mo_coeff1[:,nocc:] = numpy.dot(mo_coeff[:,nocc:], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(ncore):
                log.debug('i = %d, <i|F|i> = %12.8f', nocc+i+1, w[i])
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
        max_orb_stepsize : float
            The step size for orbital rotation.  Small step size is prefered.
            Default is 0.05.
        max_ci_stepsize : float
            The max size for approximate CI updates.  The approximate updates are
            used in 1-step algorithm, to estimate the change of CI wavefunction wrt
            the orbital rotation.  Small step size is prefered.  Default is 0.01.
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.
        max_cycle_micro : int
            Max number of micro iterations in each macro iteration.  Depending on
            systems, increasing this value might reduce the total macro
            iterations.  Generally, 2 - 3 steps should be enough.  Default is 2.
        max_cycle_micro_inner : int
            Max number of steps for the orbital rotations allowed for the augmented
            hessian solver.  It can affect the actual size of orbital rotation.
            Even with a small max_orb_stepsize, a few max_cycle_micro_inner can
            accumulate the rotation and leads to a significant change of the CAS
            space.  Depending on systems, increasing this value migh reduce the
            total number of macro iterations.  The value between 2 - 8 is preferred.
            Default is 2.
        ah_level_shift : float, for AH solver.
            Level shift for the Davidson diagonalization in AH solver.  Default is 0.
        ah_conv_tol : float, for AH solver.
            converge threshold for Davidson diagonalization in AH solver.  Default is 1e-8.
        ah_max_cycle : float, for AH solver.
            Max number of iterations allowd in AH solver.  Default is 20.
        ah_lindep : float, for AH solver.
            Linear dependence threshold for AH solver.  Default is 1e-14.
        ah_start_tol : flat, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is .5e-3.
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
            Whether to restore the natural orbital during CASSCF optimization.  Default is not.
        ci_response_space : int
            subspace size to solve the CI vector response.  Default is 2.

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
        self.max_orb_stepsize = .04
# small max_ci_stepsize is good to converge, since steepest descent is used
        self.max_ci_stepsize = .01
#TODO:self.inner_rotation = False # active-active rotation
        self.max_cycle_macro = 50
        self.max_cycle_micro = 3
        self.max_cycle_micro_inner = 3
        self.conv_tol = 1e-7
        self.conv_tol_grad = 1e-4
        # for augmented hessian
        self.ah_level_shift = 0#1e-2
        self.ah_conv_tol = 1e-8
        self.ah_max_cycle = 20
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
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-9
        self.ah_start_tol = .5e-3
        self.ah_start_cycle = 2
        self.chkfile = mf.chkfile
        self.ci_response_space = 2
        self.diis = False

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
        log.info('max. macro cycles = %d', self.max_cycle_macro)
        log.info('max. micro cycles = %d', self.max_cycle_micro)
        log.info('conv_tol = %g, (%g for gradients)', \
                 self.conv_tol, self.conv_tol_grad)
        log.info('max_cycle_micro_inner = %d', self.max_cycle_micro_inner)
        log.info('max. orb step = %g', self.max_orb_stepsize)
        log.info('max. ci step = %g', self.max_ci_stepsize)
        log.info('augmented hessian max. cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian level shift = %d', self.ah_level_shift)
        log.info('diis = %s', self.diis)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass

    def kernel(self, *args, **kwargs):
        return self.mc1step(*args, **kwargs)
    def mc1step(self, mo_coeff=None, ci0=None, macro=None, micro=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.dump_flags()

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                kernel(self, mo_coeff,
                       tol=self.conv_tol, macro=macro, micro=micro,
                       ci0=ci0, verbose=self.verbose)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo_coeff=None, ci0=None, macro=None, micro=None):
        from pyscf.mcscf import mc2step
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.mol.check_sanity(self)

        self.dump_flags()

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                mc2step.kernel(self, mo_coeff,
                               tol=self.conv_tol, macro=macro, micro=micro,
                               ci0=ci0, verbose=self.verbose)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def casci(self, mo_coeff, ci0=None, eris=None):
        if eris is None:
            fcasci = self
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)
        log = logger.Logger(self.stdout, self.verbose)
        return casci.kernel(fcasci, mo_coeff, ci0=ci0, verbose=log)

    def pack_uniq_var(self, mat):
        ncore = self.ncore
        nocc = ncore + self.ncas
        if self.frozen:
            nmo = self.mo_coeff.shape[1]
            idx = numpy.ones(nmo, dtype=numpy.bool)
            if isinstance(self.frozen, (int, numpy.integer)):
                idx[:self.frozen] = False
            else:
                idx[self.frozen] = False
            v = []
            v.append(mat[ncore:nocc,:ncore][idx[ncore:nocc,None]&idx[:ncore]])
            v.append(mat[nocc:,:nocc][idx[nocc:,None]&idx[:nocc]])
        else:
            v = []
            # active-core
            v.append(mat[ncore:nocc,:ncore].ravel())
            #TODO:if self.inner_rotation:
            #TODO:    # active-active
            #TODO:    v.append(mat[ncore:nocc,ncore:nocc].ravel())
            # virtual-core, virtual-active
            v.append(mat[nocc:,:nocc].ravel())
        return numpy.hstack(v)

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nmo = self.mo_coeff.shape[1]
        nvir = nmo - nocc
        mat = numpy.zeros((nmo,nmo))
        if self.frozen:
            idx = numpy.ones(nmo, dtype=numpy.bool)
            if isinstance(self.frozen, (int, numpy.integer)):
                idx[:self.frozen] = False
            else:
                idx[self.frozen] = False
            ncore1 = idx[:ncore].sum()
            ncas1 = idx[ncore:nocc].sum()
            nocc1 = ncore1 + ncas1
            nvir1 = idx[nocc:].sum()
            if ncore1 > 0:
                mat[ncore:nocc,:ncore][idx[ncore:nocc,None]&idx[:ncore]] = v[:ncas1*ncore1]
            if nvir1 > 0:
                mat[nocc:,:nocc][idx[nocc:,None]&idx[:nocc]] = v[-nvir1*nocc1:]
        else:
            if ncore > 0:
                mat[ncore:nocc,:ncore] = v[:ncas*ncore].reshape(ncas,ncore)
            # virtual-core, virtual-active
            if nvir > 0:
                mat[nocc:,:nocc] = v[-nvir*nocc:].reshape(nvir,nocc)
        mat[:ncore,ncore:nocc] = -mat[ncore:nocc,:ncore].T
        mat[:nocc,nocc:] = -mat[nocc:,:nocc].T
        return mat

    def gen_g_hop(self, *args):
        return gen_g_hop(self, *args)

    def rotate_orb_cc(self, mo, fcasdm1, fcasdm2, eris, verbose):
        return rotate_orb_cc(self, mo, fcasdm1, fcasdm2, eris, verbose)

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
#        eris.vhf =(numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])*2
#                  -numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore]))
#        eris.aapp = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
#        eris.appa = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
## for update_jk_in_ah
#        capp = eri[:ncore,ncore:nocc,:,:]
#        cpap = eri[:ncore,:,ncore:nocc,:]
#        ccvp = eri[:ncore,:ncore,ncore:,:]
#        cpcv = eri[:ncore,:,:ncore,ncore:]
#        cvcp = eri[:ncore,ncore:,:ncore,:]
#        cPAp = cpap * 4 - capp.transpose(0,3,1,2) - cpap.transpose(0,3,2,1)
#        cPCv = cpcv * 4 - ccvp.transpose(0,3,1,2) - cvcp.transpose(0,3,2,1)
#        eris.Iapcv = cPAp.transpose(2,3,0,1)[:,:,:,ncore:]
#        eris.Icvcv = cPCv.transpose(2,3,0,1).copy()
#        return eris

        mem = mc_ao2mo._mem_usage(self.ncore, self.ncas,
                                  self.mo_coeff.shape[1])[1]
        if mem > self.max_memory*.9:
            return mc_ao2mo._ERIS(self, mo, 'incore', 0)
        else:
            return mc_ao2mo._ERIS(self, mo, 'incore', 1)

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
        nmo = mo.shape[1]

        if hasattr(eris, 'Icvcv'):
            nvir = nmo - ncore
            vhf3c = numpy.dot(eris.Icvcv.reshape(-1,ncore*nvir),
                              r[:ncore,ncore:].ravel()).reshape(ncore,-1)
            vhf3a = numpy.einsum('uqcp,cp->uq', eris.Iapcv, r[:ncore,ncore:])
            dm4 = numpy.dot(casdm1, r[ncore:nocc])
            vhf4 = numpy.einsum('uqcp,uq->cp', eris.Iapcv, dm4)
            va = numpy.dot(casdm1, vhf3a)
            vc = 2 * vhf3c + vhf4
        else:
            dm3 = reduce(numpy.dot, (mo[:,:ncore], r[:ncore,ncore:],
                                     mo[:,ncore:].T))
            dm3 = dm3 + dm3.T
            dm4 = reduce(numpy.dot, (mo[:,ncore:nocc], casdm1, r[ncore:nocc],
                                     mo.T))
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
        nmo = mo.shape[1]
        nocc = ncore + ncas
        uc = u[:,:ncore]
        ua = u[:,ncore:nocc].copy()
        ra = rmat[:,ncore:nocc].copy()

        ddm = numpy.dot(uc, uc.T) * 2 # ~ dm(1) + dm(2)
        ddm[numpy.diag_indices(ncore)] -= 2
        jk =(reduce(numpy.dot, (ua.T, eris.vhf_c, ua)) # ~ jk(0) + jk(1) + jk(2)
           + numpy.einsum('uvpq,pq->uv', eris.aapp, ddm.T)
           - numpy.einsum('upqv,pq->uv', eris.appa, ddm) * .5)
        h1e_mo = reduce(numpy.dot, (mo.T, self.get_hcore(), mo))
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

# (@)       h2eff = self.fcisolver.absorb_h1e(h1cas, aaaa, ncas, nelecas, .5)
# (@)       g = self.fcisolver.contract_2e(h2eff, fcivec, ncas, nelecas).ravel()

        # pure core response
        # response of (1/2 dm * vhf * dm) ~ ddm*vhf
# Should I consider core response as a part of CI gradients?
        ecore =(numpy.einsum('pq,pq->', h1e_mo, ddm)
              + numpy.einsum('pq,pq->', eris.vhf_c, ddm))
        ### hessian_co part end ###

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)
        hc = self.fcisolver.contract_2e(h2eff, fcivec, ncas, nelecas).ravel()

#        ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ci0=fcivec)[1]   # (!)
# In equation (!), h1 and h2 are approximation to the fully transformed
# hamiltonain wrt rotated MO.  h1 and h2 have only 0th and part of 1st order
# (lack of the VHF[core DM response]).  Fully solving equation (!) would lead
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

        g = hc - (e_ci-ecore) * fcivec.ravel()  # hc-eci*fcivec equals to eqs. (@)
        dcmax = numpy.max(abs(g))
        if self.ci_response_space == 1 or dcmax > self.max_ci_stepsize*10:
            logger.debug(self, 'CI step by gradient descent')
# * minimal subspace is identical to simple gradient updates
            # dynamic ci_stepsize: ci graidents may change the CI vector too much,
            # which causes oscillation in macro iterations
            max_step = numpy.linalg.norm(rmat) * .1
            if self.ci_response_space == 1 and max_step < self.max_ci_stepsize:
                logger.debug1(self, 'Set CI step size to %g', max_step)
            else:
                max_step = self.max_ci_stepsize

            dc = -g.ravel()
            if dcmax > max_step:
                ci1 = fcivec.ravel() + dc * (max_step/dcmax)
            else:
                ci1 = fcivec.ravel() + dc
            ci1 *= 1/numpy.linalg.norm(ci1)

        else: # should we switch to 2D subspace when rmat is small?

# * 2D subspace spanned by fcivec and hc.  It does not have significant
#   improvement to minimal subspace.
#FIXME: trial vector in terms of Davidson precond seems worse than hc?
            #hdiag = self.fcisolver.make_hdiag(h1, h2, ncas, nelecas)
            #addr, h0 = self.fcisolver.pspace(h1, h2, ncas, nelecas, hdiag)
            #pw, pv = scipy.linalg.eigh(h0)
            #precond = self.fcisolver.make_precond(hdiag, pw, pv, addr)
            #x1 = precond(hc-fcivec.ravel()*e_ci, e_ci, fcivec.ravel())
#            x1 = hc - fcivec.ravel() * e_ci
#            hx1 = self.fcisolver.contract_2e(h2eff, x1, ncas, nelecas).ravel()
#            heff = numpy.zeros((2,2))
#            seff = numpy.zeros((2,2))
#            heff[0,0] = numpy.dot(fcivec.ravel(), hc)
#            heff[0,1] = heff[1,0] = numpy.dot(x1, hc)
#            heff[1,1] = numpy.dot(x1, hx1)
#            seff[0,0] = 1 #numpy.dot(fcivec.ravel(), fcivec.ravel())
#            seff[0,1] = seff[1,0] = numpy.dot(fcivec.ravel(), x1)
#            seff[1,1] = numpy.dot(x1, x1)
#            w, v = scipy.linalg.eigh(heff, seff)
#            ci1 = fcivec.ravel() * v[0,0] + x1.ravel() * v[1,0]
            if self.ci_response_space > 6:
                logger.debug(self, 'CI step by full response')
                # full response
                e, ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ci0=fcivec)
            else:
                nd = max(self.ci_response_space, 2)
                logger.debug(self, 'CI step by %dD subspace response', nd)
                xs = [fcivec.ravel()]
                ax = [hc]
                heff = numpy.empty((nd,nd))
                seff = numpy.empty((nd,nd))
                heff[0,0] = numpy.dot(xs[0], ax[0])
                seff[0,0] = 1
                for i in range(1, nd):
                    xs.append(ax[i-1] - xs[i-1] * (e_ci-ecore))
                    ax.append(self.fcisolver.contract_2e(h2eff, xs[i], ncas,
                                                         nelecas).ravel())
                    for j in range(i+1):
                        heff[i,j] = heff[j,i] = numpy.dot(xs[i], ax[j])
                        seff[i,j] = seff[j,i] = numpy.dot(xs[i], xs[j])
                e, v = scipy.linalg.eigh(heff, seff)
                ci1 = 0
                for i in range(nd):
                    ci1 += xs[i] * v[i,0]

        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, ncas, nelecas)

        return casdm1, casdm2, g

    def save_mo_coeff(self, mo_coeff, *args):
        pyscf.scf.chkfile.dump(self.chkfile, 'mcscf/mo_coeff', mo_coeff)
    def load_mo_coeff(self):
        return pyscf.scf.chkfile.load(self.chkfile, 'mcscf/mo_coeff')

    def get_jk(self, mol, dm, hermi=1):
        return self._scf.get_jk(mol, dm, hermi=1)

    def dump_chk(self, *args, **kwargs):
        from pyscf.mcscf import chkfile
        chkfile.dump_mcscf(self.mol, self.chkfile, *args, **kwargs)

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
    print(emc - -15.950852049859)


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
