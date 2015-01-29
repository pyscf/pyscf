#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import copy
import tempfile
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.scf
from pyscf.mcscf import casci_uhf
from pyscf.mcscf import mc1step
from pyscf.mcscf import aug_hessian
from pyscf.mcscf import mc_ao2mo_uhf

#FIXME:  when the number of core orbitals are different for alpha and beta,
# the convergence are very unstable and slow

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, casdm1s, casdm2s, eris):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = (ncas + ncore[0], ncas + ncore[1])
    nmo = casscf.mo_coeff[0].shape[1]

    dm1 = numpy.zeros((2,nmo,nmo))
    idx = numpy.arange(ncore[0])
    dm1[0,idx,idx] = 1
    idx = numpy.arange(ncore[1])
    dm1[1,idx,idx] = 1
    dm1[0,ncore[0]:nocc[0],ncore[0]:nocc[0]] = casdm1s[0]
    dm1[1,ncore[1]:nocc[1],ncore[1]:nocc[1]] = casdm1s[1]

    # part2, part3
    vhf_c = (numpy.einsum('ipq->pq', eris.jkcpp) + eris.jC_pp,
             numpy.einsum('ipq->pq', eris.jkcPP) + eris.jc_PP)
    vhf_ca = (vhf_c[0] + numpy.einsum('uvpq,uv->pq', eris.aapp, casdm1s[0]) \
                       - numpy.einsum('upqv,uv->pq', eris.appa, casdm1s[0]) \
                       + numpy.einsum('uvpq,uv->pq', eris.AApp, casdm1s[1]),
              vhf_c[1] + numpy.einsum('uvpq,uv->pq', eris.aaPP, casdm1s[0]) \
                       + numpy.einsum('uvpq,uv->pq', eris.AAPP, casdm1s[1]) \
                       - numpy.einsum('upqv,uv->pq', eris.APPA, casdm1s[1]),)

    ################# gradient #################
    hdm2 = [ numpy.einsum('tuvw,vwpq->tupq', casdm2s[0], eris.aapp) \
           + numpy.einsum('tuvw,vwpq->tupq', casdm2s[1], eris.AApp),
             numpy.einsum('vwtu,vwpq->tupq', casdm2s[1], eris.aaPP) \
           + numpy.einsum('tuvw,vwpq->tupq', casdm2s[2], eris.AAPP)]

    hcore = casscf.get_hcore()
    h1e_mo = (reduce(numpy.dot, (mo[0].T, hcore[0], mo[0])),
              reduce(numpy.dot, (mo[1].T, hcore[1], mo[1])))
    g = [numpy.dot(h1e_mo[0], dm1[0]),
         numpy.dot(h1e_mo[1], dm1[1])]
    def gpart(m):
        g[m][:,:ncore[m]] += vhf_ca[m][:,:ncore[m]]
        g[m][:,ncore[m]:nocc[m]] += \
                numpy.einsum('vuuq->qv', hdm2[m][:,:,ncore[m]:nocc[m]]) \
              + numpy.dot(vhf_c[m][:,ncore[m]:nocc[m]], casdm1s[m])
    gpart(0)
    gpart(1)

    ############## hessian, diagonal ###########
    # part1
    tmp = casdm2s[0].transpose(1,2,0,3) + casdm2s[0].transpose(0,2,1,3)
    hdm2apap = numpy.einsum('uvtw,tpqw->upvq', tmp, eris.appa)
    hdm2apap += hdm2[0].transpose(0,2,1,3)
    hdm2[0] = hdm2apap

    tmp = casdm2s[1].transpose(1,2,0,3) + casdm2s[1].transpose(0,2,1,3)
    hdm2apAP = numpy.einsum('uvtw,tpqw->upvq', tmp, eris.apPA) # (jp|RK) *[e(jq,SK) + e(jq,LS)] => qSpR
    # (JP|rk) *[e(sk,JQ) + e(ls,JQ)] => QsPr
    hdm2APap = hdm2apAP.transpose(2,3,0,1)

    tmp = casdm2s[2].transpose(1,2,0,3) + casdm2s[2].transpose(0,2,1,3)
    hdm2APAP = numpy.einsum('uvtw,tpqw->upvq', tmp, eris.APPA)
    hdm2APAP += hdm2[1].transpose(0,2,1,3)
    hdm2[1] = hdm2APAP

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = [numpy.einsum('ii,jj->ij', h1e_mo[0], dm1[0]) - h1e_mo[0] * dm1[0],
              numpy.einsum('ii,jj->ij', h1e_mo[1], dm1[1]) - h1e_mo[1] * dm1[1]]
    h_diag[0] = h_diag[0] + h_diag[0].T
    h_diag[1] = h_diag[1] + h_diag[1].T

    # part8
    idx = numpy.arange(nmo)
    g_diag = g[0].diagonal()
    h_diag[0] -= g_diag + g_diag.reshape(-1,1)
    h_diag[0][idx,idx] += g_diag * 2
    g_diag = g[1].diagonal()
    h_diag[1] -= g_diag + g_diag.reshape(-1,1)
    h_diag[1][idx,idx] += g_diag * 2

    # part2, part3
    def fpart2(m):
        v_diag = vhf_ca[m].diagonal() # (pr|kl) * e(sq,lk)
        h_diag[m][:,:ncore[m]] += v_diag.reshape(-1,1)
        h_diag[m][:ncore[m]] += v_diag
        idx = numpy.arange(ncore[m])
        # (V_{qr} delta_{ps} + V_{ps} delta_{qr}) delta_{pr} delta_{sq}
        h_diag[m][idx,idx] -= v_diag[:ncore[m]] * 2
    fpart2(0)
    fpart2(1)

    def fpart3(m):
        # V_{pr} e_{sq}
        tmp = numpy.einsum('ii,jj->ij', vhf_c[m], casdm1s[m])
        h_diag[m][:,ncore[m]:nocc[m]] += tmp
        h_diag[m][ncore[m]:nocc[m],:] += tmp.T
        tmp = -vhf_c[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] * casdm1s[m]
        h_diag[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] += tmp + tmp.T
    fpart3(0)
    fpart3(1)

    # part4
    def fpart4(jkcpp, m):
        # (qp|rs)-(pr|sq) rp in core
        tmp = -numpy.einsum('cpp->cp', jkcpp)
        # (qp|sr) - (qr|sp) rp in core => 0
        h_diag[m][:ncore[m],:] += tmp
        h_diag[m][:,:ncore[m]] += tmp.T
        h_diag[m][:ncore[m],:ncore[m]] -= tmp[:,:ncore[m]] * 2
    fpart4(eris.jkcpp, 0)
    fpart4(eris.jkcPP, 1)

    # part5 and part6 diag
    #+(qr|kp) e_s^k  p in core, sk in active
    #+(qr|sl) e_l^p  s in core, pl in active
    #-(qj|sr) e_j^p  s in core, jp in active
    #-(qp|kr) e_s^k  p in core, sk in active
    #+(qj|rs) e_j^p  s in core, jp in active
    #+(qp|rl) e_l^s  p in core, ls in active
    #-(qs|rl) e_l^p  s in core, lp in active
    #-(qj|rp) e_j^s  p in core, js in active
    def fpart5(jkcpp, m):
        jkcaa = jkcpp[:,ncore[m]:nocc[m],ncore[m]:nocc[m]]
        tmp = -2 * numpy.einsum('jik,ik->ji', jkcaa, casdm1s[m])
        h_diag[m][:ncore[m],ncore[m]:nocc[m]] -= tmp
        h_diag[m][ncore[m]:nocc[m],:ncore[m]] -= tmp.T
    fpart5(eris.jkcpp, 0)
    fpart5(eris.jkcPP, 1)

    def fpart1(m):
        v_diag = numpy.einsum('ijij->ij', hdm2[m])
        h_diag[m][ncore[m]:nocc[m],:] += v_diag
        h_diag[m][:,ncore[m]:nocc[m]] += v_diag.T
    fpart1(0)
    fpart1(1)

    g_orb = casscf.pack_uniq_var((g[0]-g[0].T, g[1]-g[1].T))
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1a, x1b = casscf.unpack_uniq_var(x)
        xa_cu = x1a[:ncore[0],ncore[0]:]
        xa_av = x1a[ncore[0]:nocc[0],nocc[0]:]
        xa_ac = x1a[ncore[0]:nocc[0],:ncore[0]]
        xb_cu = x1b[:ncore[1],ncore[1]:]
        xb_av = x1b[ncore[1]:nocc[1],nocc[1]:]
        xb_ac = x1b[ncore[1]:nocc[1],:ncore[1]]

        # part7
        x2a = reduce(numpy.dot, (h1e_mo[0], x1a, dm1[0]))
        x2b = reduce(numpy.dot, (h1e_mo[1], x1b, dm1[1]))
        # part8, the hessian gives
        #x2a -= numpy.dot(g[0], x1a)
        #x2b -= numpy.dot(g[1], x1b)
        # it may ruin the hermitian of hessian unless g == g.T. So symmetrize it
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        x2a -= numpy.dot(g[0]+g[0].T, x1a) * .5
        x2b -= numpy.dot(g[1]+g[1].T, x1b) * .5
        # part2
        x2a[:ncore[0]] += numpy.dot(xa_cu, vhf_ca[0][ncore[0]:])
        x2b[:ncore[1]] += numpy.dot(xb_cu, vhf_ca[1][ncore[1]:])
        # part3
        def fpart3(m, x2, x_av, x_ac):
            x2[ncore[m]:nocc[m]] += reduce(numpy.dot, (casdm1s[m], x_av, vhf_c[m][nocc[m]:])) \
                                  + reduce(numpy.dot, (casdm1s[m], x_ac, vhf_c[m][:ncore[m]]))
        fpart3(0, x2a, xa_av, xa_ac)
        fpart3(1, x2b, xb_av, xb_ac)
        # part4, part5, part6
        if ncore[0] > 0 or ncore[1] > 0:
            va, vc = casscf.update_jk_in_ah(mo, (x1a,x1b), casdm1s, eris)
            x2a[ncore[0]:nocc[0]] += va[0]
            x2b[ncore[1]:nocc[1]] += va[1]
            x2a[:ncore[0],ncore[0]:] += vc[0]
            x2b[:ncore[1],ncore[1]:] += vc[1]

        # part1
        x2a[ncore[0]:nocc[0]] += numpy.einsum('upvr,vr->up', hdm2apap, x1a[ncore[0]:nocc[0]])
        x2a[ncore[0]:nocc[0]] += numpy.einsum('upvr,vr->up', hdm2apAP, x1b[ncore[1]:nocc[1]])
        x2b[ncore[1]:nocc[1]] += numpy.einsum('vrup,vr->up', hdm2apAP, x1a[ncore[0]:nocc[0]])
        x2b[ncore[1]:nocc[1]] += numpy.einsum('upvr,vr->up', hdm2APAP, x1b[ncore[1]:nocc[1]])

        x2a = x2a - x2a.T
        x2b = x2b - x2b.T
        return casscf.pack_uniq_var((x2a,x2b))
    return g_orb, h_op, h_diag


def rotate_orb_ah(casscf, mo, casdm1s, casdm2s, eris, dx=0, verbose=None):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)

    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nmo = mo[0].shape[1]

    t2m = (time.clock(), time.time())
    g_orb0, h_op, h_diag = casscf.gen_g_hop(mo, casdm1s, casdm2s, eris)
    t3m = log.timer('gen h_op', *t2m)

    precond = lambda x, e: x/(h_diag-(e-casscf.ah_level_shift))
    u = (numpy.eye(nmo),) * 2

    if isinstance(dx, int):
        x0 = g_orb0
        g_orb = g_orb0
    else:
        x0 = dx
        g_orb = g_orb0 + h_op(dx)

    norm_gprev = numpy.linalg.norm(g_orb)
    ah_start_tol = min(norm_gprev, casscf.ah_start_tol)
    g_op = lambda: g_orb
    imic = 0
    for ihop, w, dxi in aug_hessian.davidson_cc(h_op, g_op, precond, x0, log,
                                                tol=casscf.ah_conv_tol,
                                                start_tol=ah_start_tol,
                                                max_cycle=casscf.ah_max_cycle,
                                                max_stepsize=1.5,
                                                lindep=casscf.ah_lindep,
                                                start_cycle=casscf.ah_start_cycle):
        imic += 1
        dx1 = dxi
        dxmax = numpy.max(abs(dx1))
        if dxmax > casscf.max_orb_stepsize:
            dx1 = dx1 * (casscf.max_orb_stepsize/dxmax)
        dx = dx + dx1
        dr = casscf.unpack_uniq_var(dx1)
        u = list(map(numpy.dot, u, map(mc1step.expmat, dr)))

# within few steps, g_orb + \sum_i h_op(dx_i) is a good approximation to the
# exact gradients. After few updates, decreasing the approx gradients may
# result in the increase of the real gradient.
        g_orb = g_orb0 + h_op(dx)
        norm_gorb = numpy.linalg.norm(g_orb)
        norm_dx1 = numpy.linalg.norm(dx1)
        log.debug('    inner iter %d, |g[o]|=%4.3g, |dx|=%4.3g, max(|x|)=%4.3g, eig=%4.3g',
                   imic, norm_gorb, norm_dx1, dxmax, w)
        if imic >= casscf.max_cycle_micro_inner \
           or norm_gorb > norm_gprev \
           or norm_gorb < casscf.conv_tol_grad:
            break
        norm_gprev = numpy.linalg.norm(g_orb)

    t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)
    return u, dx, g_orb, imic+ihop

# dc = h_{co} * dr
def hessian_co(casscf, mo, rmat, fcivec, e_ci, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nmo = mo[0].shape[1]
    nocc = (ncas + ncore[0], ncas + ncore[1])
    mocc = (mo[0][:,:nocc[0]], mo[1][:,:nocc[1]])

    hcore = casscf.get_hcore()
    h1effa = reduce(numpy.dot, (rmat[0][:,:nocc[0]].T, mo[0].T, hcore[0], mo[0][:,:nocc[0]]))
    h1effb = reduce(numpy.dot, (rmat[1][:,:nocc[1]].T, mo[1].T, hcore[1], mo[1][:,:nocc[1]]))
    h1effa = h1effa + h1effa.T
    h1effb = h1effb + h1effb.T

    aapc = eris.aapp[:,:,:,:ncore[0]]
    aaPC = eris.aaPP[:,:,:,:ncore[1]]
    AApc = eris.AApp[:,:,:,:ncore[0]]
    AAPC = eris.AAPP[:,:,:,:ncore[1]]
    apca = eris.appa[:,:,:ncore[0],:]
    APCA = eris.APPA[:,:,:ncore[1],:]
    jka = numpy.einsum('iup->up', eris.jkcpp[:,:nocc[0]]) + eris.jC_pp[:nocc[0]]
    v1a = numpy.einsum('up,pv->uv', jka[ncore[0]:], rmat[0][:,ncore[0]:nocc[0]]) \
        + numpy.einsum('uvpi,pi->uv', aapc-apca.transpose(0,3,1,2), rmat[0][:,:ncore[0]]) \
        + numpy.einsum('uvpi,pi->uv', aaPC, rmat[1][:,:ncore[1]])
    jkb = numpy.einsum('iup->up', eris.jkcPP[:,:nocc[1]]) + eris.jc_PP[:nocc[1]]
    v1b = numpy.einsum('up,pv->uv', jkb[ncore[1]:], rmat[1][:,ncore[1]:nocc[1]]) \
        + numpy.einsum('uvpi,pi->uv', AApc, rmat[0][:,:ncore[0]]) \
        + numpy.einsum('uvpi,pi->uv', AAPC-APCA.transpose(0,3,1,2), rmat[1][:,:ncore[1]])
    h1cas = (h1effa[ncore[0]:,ncore[0]:] + (v1a + v1a.T),
             h1effb[ncore[1]:,ncore[1]:] + (v1b + v1b.T))

    aaap = eris.aapp[:,:,ncore[0]:nocc[0],:]
    aaAP = eris.aaPP[:,:,ncore[1]:nocc[1],:]
    AAap = eris.AApp[:,:,ncore[1]:nocc[1],:]
    AAAP = eris.AAPP[:,:,ncore[1]:nocc[1],:]
    aaaa = numpy.einsum('tuvp,pw->tuvw', aaap, rmat[0][:,ncore[0]:nocc[0]])
    aaaa = aaaa + aaaa.transpose(0,1,3,2)
    aaaa = aaaa + aaaa.transpose(2,3,0,1)
    AAAA = numpy.einsum('tuvp,pw->tuvw', AAAP, rmat[1][:,ncore[1]:nocc[1]])
    AAAA = AAAA + AAAA.transpose(0,1,3,2)
    AAAA = AAAA + AAAA.transpose(2,3,0,1)
    tmp = (numpy.einsum('vwtp,pu->tuvw', AAap, rmat[0][:,ncore[0]:nocc[0]]),
           numpy.einsum('tuvp,pw->tuvw', aaAP, rmat[1][:,ncore[1]:nocc[1]]))
    aaAA = tmp[0] + tmp[0].transpose(1,0,2,3) \
         + tmp[1] + tmp[1].transpose(0,1,3,2)
    h2eff = casscf.fcisolver.absorb_h1e(h1cas, (aaaa,aaAA,AAAA), ncas, nelecas, .5)
    hc = casscf.fcisolver.contract_2e(h2eff, fcivec, ncas, nelecas).ravel()

    # pure core response
    ecore = h1effa[:ncore[0]].trace() + h1effb[:ncore[1]].trace() \
          + numpy.einsum('jp,pj->', jka[:ncore[0]], rmat[0][:,:ncore[0]])*2 \
          + numpy.einsum('jp,pj->', jkb[:ncore[1]], rmat[1][:,:ncore[1]])*2
    hc += ecore * fcivec.ravel()
    return hc


def kernel(casscf, mo_coeff, tol=1e-7, macro=30, micro=8, \
           ci0=None, verbose=None, **cikwargs):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 1-step CASSCF')

    mo = mo_coeff
    nmo = mo[0].shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = (ncas + ncore[0], ncas + ncore[1])
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.update_ao2mo(mo)
    e_tot, e_ci, fcivec = casscf.casci(mo, ci0, eris, **cikwargs)
    log.info('CASCI E = %.15g', e_tot)
    if ncas == nmo:
        return e_tot, e_ci, fcivec, mo
    elast = e_tot
    conv = False
    toloose = casscf.conv_tol_grad
    totmicro = totinner = 0
    imicro = 0
    norm_gorb = norm_gci = 0
    casdm1_old = (0, 0)

    t2m = t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    for imacro in range(macro):
        casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, ncas, casscf.nelecas)
        u, dx, g_orb, ninner = casscf.rotate_orb(mo, casdm1, casdm2, eris, 0)
        norm_gorb = numpy.linalg.norm(g_orb)
        norm_dm1 = numpy.linalg.norm(casdm1[0] - casdm1_old[0]) \
                 + numpy.linalg.norm(casdm1[1] - casdm1_old[1])
        t3m = log.timer('CAS DM + orbital rotation', *t2m)
        totmicro += 1

        casdm1_old = casdm1
        for imicro in range(micro):
# approximate newton step, fcivec is not updated during micro iters
            casdm1, casdm2, gci = casscf.update_casdm(mo, u, fcivec, e_ci, eris)
            norm_gci = numpy.linalg.norm(gci)
            norm_dm1 = numpy.linalg.norm(casdm1[0] - casdm1_old[0]) \
                     + numpy.linalg.norm(casdm1[1] - casdm1_old[1])
            t3m = log.timer('update CAS DM', *t3m)

            u1, dx, g_orb, nin = casscf.rotate_orb(mo, casdm1, casdm2, eris, dx)
            u = list(map(numpy.dot, u, u1))
            ninner += nin
            t3m = log.timer('orbital rotation', *t3m)
            norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
            norm_gorb = numpy.linalg.norm(g_orb)
            totmicro += 1

            log.debug('micro %d, e_ci = %.12g, |u-1|=%4.3g, |g[o]|=%4.3g, ' \
                      '|g[c]|=%4.3g, |dm1|=%4.3g',
                      imicro, e_ci, norm_t, norm_gorb, norm_gci, norm_dm1)
            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if (norm_t < toloose or norm_gorb < toloose or
                norm_gci < toloose or norm_dm1 < toloose):
                if casscf.natorb:
                    natocc, natorb = scipy.linalg.eigh(-casdm1[0])
                    u[0][:,ncore[0]:nocc[0]] = \
                            numpy.dot(u[0][:,ncore[0]:nocc[0]], natorb)
                    log.debug1('natural occ alpha = %s', str(natocc))
                    natocc, natorb = scipy.linalg.eigh(-casdm1[1])
                    u[1][:,ncore[1]:nocc[1]] = \
                            numpy.dot(u[1][:,ncore[1]:nocc[1]], natorb)
                    log.debug1('natural occ beta = %s', str(natocc))
                break

        totinner += ninner

        mo = list(map(numpy.dot, mo, u))
        casscf.save_mo_coeff(mo, imacro, imicro)

        eris = None # to avoid using too much memory
        eris = casscf.update_ao2mo(mo)
        t3m = log.timer('update eri', *t3m)

        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris, **cikwargs)
        log.info('macro iter %d (%d ah, %d micro), CASSCF E = %.15g, dE = %.8g,',
                 imacro, ninner, imicro+1, e_tot, e_tot-elast)
        log.info('               |grad[o]|=%4.3g, |grad[c]|=%4.3g, |dm1|=%4.3g',
                 norm_gorb, norm_gci, norm_dm1)
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        if abs(e_tot - elast) < tol \
           and (norm_gorb < toloose and norm_dm1 < toloose):
            conv = True
            break
        else:
            elast = e_tot

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.log('1-step CASSCF, energy = %.15g', e_tot)
    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_ci, fcivec, mo


class CASSCF(casci_uhf.CASCI):
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        casci_uhf.CASCI.__init__(self, mol, mf, ncas, nelecas, ncore)
# the max orbital rotation and CI increment, prefer small step size
        self.max_orb_stepsize = .03
# small max_ci_stepsize is good to converge, since steepest descent is used
        self.max_ci_stepsize = .01
#TODO:self.inner_rotation = False # active-active rotation
        self.max_cycle_macro = 50
        self.max_cycle_micro = 2
# num steps to approx orbital rotation without integral transformation.
# Increasing steps do not help converge since the approx gradient might be
# very diff to real gradient after few steps. If the step predicted by AH is
# good enough, it can be set to 1 or 2 steps.
        self.max_cycle_micro_inner = 4
        self.conv_tol = 1e-7
        self.conv_tol_grad = 1e-4
        # for augmented hessian
        self.ah_level_shift = 0#1e-2
        self.ah_conv_tol = 1e-8
        self.ah_max_cycle = 15
        self.ah_lindep = self.ah_conv_tol**2
        self.ah_start_tol = 1e-4
        self.ah_start_cycle = 3
        self.chkfile = mf.chkfile
        self.natorb = False

        self.fcisolver.max_cycle = 50
        self.e_tot = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.converged = False

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** UHF-CASSCF flags ********')
        ncore = self.ncore
        nmo = self.mo_coeff[0].shape[1]
        nvir_alpha = nmo - self.ncore[0] - self.ncas
        nvir_beta  = nmo - self.ncore[1]  - self.ncas
        log.info('CAS (%de+%de, %do), ncore = [%d+%d], nvir = [%d+%d]', \
                 self.nelecas[0], self.nelecas[1], self.ncas,
                 self.ncore[0], self.ncore[1], nvir_alpha, nvir_beta)
        if self.ncore[0] != self.ncore[1]:
            log.warn('converge might be slow since num alpha core %d != num beta core %d',
                     self.ncore[0], self.ncore[1])
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
        log.info('max_memory %d MB', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except:
            pass

    def kernel(self, *args, **kwargs):
        return self.mc1step(*args, **kwargs)
    def mc1step(self, mo_coeff=None, ci0=None, macro=None, micro=None, **cikwargs):
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
                kernel(self, mo_coeff, \
                       tol=self.conv_tol, macro=macro, micro=micro, \
                       ci0=ci0, verbose=self.verbose, **cikwargs)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo_coeff=None, ci0=None, macro=None, micro=None, **cikwargs):
        from pyscf.mcscf import mc2step_uhf
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
                mc2step_uhf.kernel(self, mo_coeff, \
                                   tol=self.conv_tol, macro=macro, micro=micro, \
                                   ci0=ci0, verbose=self.verbose, **cikwargs)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def casci(self, mo_coeff, ci0=None, eris=None, **cikwargs):
        if eris is None:
            fcasci = self
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)
        return casci_uhf.kernel(fcasci, mo_coeff, ci0=ci0, verbose=0, **cikwargs)

    def pack_uniq_var(self, mat):
        v = []

        # alpha
        ncore = self.ncore[0]
        nocc = ncore + self.ncas
        # active-core
        v.append(mat[0][ncore:nocc,:ncore].ravel())
        # alpha virtual-core, virtual-active
        v.append(mat[0][nocc:,:nocc].ravel())

        # beta
        ncore = self.ncore[1]
        nocc = ncore + self.ncas
        v.append(mat[1][ncore:nocc,:ncore].ravel())
        v.append(mat[1][nocc:,:nocc].ravel())
        return numpy.hstack(v)

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff[0].shape[1]

        # alpha
        ncore = self.ncore[0]
        ncas = self.ncas
        nocc = ncore + self.ncas
        nvir = nmo - nocc
        mata = numpy.zeros((nmo,nmo))
        if ncore > 0:
            mata[ncore:nocc,:ncore] = v[:ncas*ncore].reshape(ncas,-1)
            mata[:ncore,ncore:nocc] = -mata[ncore:nocc,:ncore].T
        if nvir > 0:
            mata[nocc:,:nocc] = v[ncas*ncore:ncas*ncore+nvir*nocc].reshape(nvir,-1)
            mata[:nocc,nocc:] = -mata[nocc:,:nocc].T
        v = v[ncas*ncore+nvir*nocc:]

        # beta
        ncore = self.ncore[1]
        ncas = self.ncas
        nocc = ncore + self.ncas
        nvir = nmo - nocc
        matb = numpy.zeros((nmo,nmo))
        if ncore > 0:
            matb[ncore:nocc,:ncore] = v[:ncas*ncore].reshape(ncas,-1)
            matb[:ncore,ncore:nocc] = -matb[ncore:nocc,:ncore].T
        if nvir > 0:
            matb[nocc:,:nocc] = v[ncas*ncore:ncas*ncore+nvir*nocc].reshape(nvir,-1)
            matb[:nocc,nocc:] = -matb[nocc:,:nocc].T
        return (mata, matb)

    def gen_g_hop(self, *args):
        return gen_g_hop(self, *args)

    def rotate_orb(self, mo, casdm1, casdm2, eris, dx=0):
        return rotate_orb_ah(self, mo, casdm1, casdm2, eris, dx, self.verbose)

    def update_ao2mo(self, mo):
#        nmo = mo[0].shape[1]
#        ncore = self.ncore
#        ncas = self.ncas
#        nocc = (ncas + ncore[0], ncas + ncore[1])
#        eriaa = pyscf.ao2mo.incore.full(self._scf._eri, mo[0])
#        eriab = pyscf.ao2mo.incore.general(self._scf._eri, (mo[0],mo[0],mo[1],mo[1]))
#        eribb = pyscf.ao2mo.incore.full(self._scf._eri, mo[1])
#        eriaa = pyscf.ao2mo.restore(1, eriaa, nmo)
#        eriab = pyscf.ao2mo.restore(1, eriab, nmo)
#        eribb = pyscf.ao2mo.restore(1, eribb, nmo)
#        eris = lambda:None
#        eris.jkcpp = numpy.einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
#                   - numpy.einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
#        eris.jkcPP = numpy.einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
#                   - numpy.einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
#        eris.jC_pp = numpy.einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
#        eris.jc_PP = numpy.einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
#        eris.aapp = numpy.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
#        eris.aaPP = numpy.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
#        eris.AApp = numpy.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
#        eris.AAPP = numpy.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
#        eris.appa = numpy.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
#        eris.apPA = numpy.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
#        eris.APPA = numpy.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])
#
#        eris.cvCV = numpy.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
#        eris.Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
#                   - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
#                   - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
#        eris.ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
#                   - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
#                   - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)
#
#        eris.Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
#                   - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
#                   - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
#        eris.IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
#                   - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
#                   - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
#        eris.apCV = numpy.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
#        eris.APcv = numpy.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))
#        return eris
        return mc_ao2mo_uhf._ERIS(self, mo)

    def update_jk_in_ah(self, mo, r, casdm1s, eris):
        ncas = self.ncas
        ncore = self.ncore
        nocc = (ncas + ncore[0], ncas + ncore[1])
        nmo = mo[0].shape[1]
        ra, rb = r
        vhf3ca = numpy.einsum('srqp,sr->qp', eris.Icvcv, ra[:ncore[0],ncore[0]:])
        vhf3ca += numpy.einsum('qpsr,sr->qp', eris.cvCV, rb[:ncore[1],ncore[1]:]) * 2
        vhf3cb = numpy.einsum('srqp,sr->qp', eris.ICVCV, rb[:ncore[1],ncore[1]:])
        vhf3cb += numpy.einsum('srqp,sr->qp', eris.cvCV, ra[:ncore[0],ncore[0]:]) * 2

        vhf3aa = numpy.einsum('kpsr,sr->kp', eris.Iapcv, ra[:ncore[0],ncore[0]:])
        vhf3aa += numpy.einsum('kpsr,sr->kp', eris.apCV, rb[:ncore[1],ncore[1]:]) * 2
        vhf3ab = numpy.einsum('kpsr,sr->kp', eris.IAPCV, rb[:ncore[1],ncore[1]:])
        vhf3ab += numpy.einsum('kpsr,sr->kp', eris.APcv, ra[:ncore[0],ncore[0]:]) * 2

        dm4 = (numpy.dot(casdm1s[0], ra[ncore[0]:nocc[0]]),
               numpy.dot(casdm1s[1], rb[ncore[1]:nocc[1]]))
        vhf4a = numpy.einsum('krqp,kr->qp', eris.Iapcv, dm4[0])
        vhf4a += numpy.einsum('krqp,kr->qp', eris.APcv, dm4[1]) * 2
        vhf4b = numpy.einsum('krqp,kr->qp', eris.IAPCV, dm4[1])
        vhf4b += numpy.einsum('krqp,kr->qp', eris.apCV, dm4[0]) * 2

        va = (numpy.dot(casdm1s[0], vhf3aa), numpy.dot(casdm1s[1], vhf3ab))
        vc = (vhf3ca + vhf4a, vhf3cb + vhf4b)
        return va, vc

    def update_casdm(self, mo, u, fcivec, e_ci, eris):
        nmo = mo[0].shape[1]
        rmat = u - numpy.eye(nmo)
        g = hessian_co(self, mo, rmat, fcivec, e_ci, eris)

        dc = -g.ravel()
        dcmax = numpy.max(abs(dc))
        max_step = min(numpy.linalg.norm(rmat)/rmat.shape[0], self.max_ci_stepsize)
        if dcmax > max_step:
            ci1 = fcivec.ravel() + dc * (max_step/dcmax)
        else:
            ci1 = fcivec.ravel() + dc
        ci1 *= 1/numpy.linalg.norm(ci1)
        casdm1, casdm2 = self.fcisolver.make_rdm12s(ci1, self.ncas, self.nelecas)
        return casdm1, casdm2, g

    def save_mo_coeff(self, mo_coeff, *args):
        pyscf.scf.chkfile.dump(self.chkfile, 'mcscf/mo_coeff', mo_coeff)


# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    # vhf for core density matrix
    s = mc._scf.get_ovlp()
    mo_inv = (numpy.dot(mo[0].T, s), numpy.dot(mo[1].T, s))
    vjk =(numpy.einsum('ipq->pq', eris.jkcpp) + eris.jC_pp,
          numpy.einsum('ipq->pq', eris.jkcPP) + eris.jc_PP)
    vhf =(reduce(numpy.dot, (mo_inv[0].T, vjk[0], mo_inv[0])),
          reduce(numpy.dot, (mo_inv[1].T, vjk[1], mo_inv[1])))
    mc.get_veff = lambda *args: vhf

    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = (ncas + ncore[0], ncas + ncore[1])
    eri_cas = (eris.aapp[:,:,ncore[0]:nocc[0],ncore[0]:nocc[0]].copy(), \
               eris.aaPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].copy(),
               eris.AAPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].copy())
    mc.ao2mo = lambda *args: eri_cas
    return mc


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
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

    mol.basis = {'H': 'sto-3g'}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = CASSCF(mol, m, 4, (2,1))
    #mo = m.mo_coeff
    mo = addons.sort_mo(mc, m.mo_coeff, [(3,4,5,6),(3,4,6,7)], 1)
    emc = kernel(mc, mo, micro=4, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -2.9782774463926618)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = CASSCF(mol, m, 4, (2,1))
    mc.verbose = 4
    #mo = m.mo_coeff
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-75.631870606190233, -75.573930418500652, 0.057940187689581535
    print(emc - -75.573930418500652, emc - -75.648547447838951)
