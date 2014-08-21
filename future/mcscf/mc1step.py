#!/usr/bin/env python
#
# File: mc1step.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
import casci
import aug_hessian
import pyscf.future.fci.direct_spin0 as fci_direct
from pyscf.lib import _mcscf

# ref. JCP, 82, 5053;  JCP, 73, 2342

def h1e_for_cas(mol, casscf, mo, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    if ncore == 0:
        vhf_c = 0
    else:
        jc_aa = eris['jc_pp'][:,ncore:nocc,ncore:nocc]
        kc_aa = eris['kc_pp'][:,ncore:nocc,ncore:nocc]
        vhf_c = numpy.einsum('iuv->uv', jc_aa) * 2 \
              - numpy.einsum('iuv->uv', kc_aa)
    h1eff = eris['h1e_mo'][ncore:nocc,ncore:nocc] + vhf_c
    return h1eff

def expmat(a):
    x1 = numpy.dot(a,a)
    u = numpy.eye(a.shape[0]) + a + .5 * x1
    x2 = numpy.dot(x1,a)
    u = u + 1./6 * x2
    #x3 = numpy.dot(x1,x1)
    #u = u + 1./24 * x3
    u, w, vh = numpy.linalg.svd(u)
    return numpy.dot(u,vh)

# gradients, hessian operator and hessian diagonal
def gen_g_hop(mol, casscf, mo, casdm1, casdm2, eris):
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    # part2, part3
    #vhf_c = numpy.einsum('ccpq->pq', eris['copp'][:ncore,:ncore]) * 2 \
    #      - numpy.einsum('cpqc->pq', eris['cppo'][:ncore,:,:,:ncore])
    vhf_c = numpy.einsum('ipq->pq', eris['jc_pp']) * 2 \
          - numpy.einsum('ipq->pq', eris['kc_pp'])
    vhf_ca = vhf_c + numpy.einsum('uvpq,uv->pq', eris['aapp'], casdm1) \
                   - numpy.einsum('upqv,uv->pq', eris['appa'], casdm1) * .5

    ################# gradient #################
    hdm2 = numpy.dot(casdm2.reshape(ncas*ncas,-1), \
                     eris['aapp'].reshape(ncas*ncas,-1)).reshape(ncas,ncas,nmo,nmo)

    g = numpy.dot(eris['h1e_mo'], dm1)
    g[:,:ncore] += vhf_ca[:,:ncore] * 2
    g[:,ncore:nocc] += numpy.einsum('vuuq->qv',hdm2[:,:,ncore:nocc]) \
            + numpy.dot(vhf_c[:,ncore:nocc],casdm1)

    ############## hessian, diagonal ###########
    # part1
    tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3)
    tmp = numpy.dot(tmp.reshape(ncas*ncas,-1), \
                    eris['appa'].transpose(0,3,1,2).reshape(-1,nmo*nmo))
    tmp = tmp.reshape(ncas,ncas,nmo,nmo)
    hdm2 = numpy.array((hdm2+tmp).transpose(0,2,1,3), order='C')

    # part7
    h_diag = numpy.einsum('ii,jj->ij', eris['h1e_mo'], dm1) - eris['h1e_mo'] * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    idx = numpy.arange(nmo)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal()
    h_diag[:,:ncore] += v_diag.reshape(-1,1) * 2
    h_diag[:ncore] += v_diag * 2
    idx = numpy.arange(ncore)
    h_diag[idx,idx] -= v_diag[:ncore] * 4
    tmp = numpy.einsum('ii,jj->ij', vhf_c, casdm1)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -vhf_c[ncore:nocc,ncore:nocc] * casdm1
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    jc_c = numpy.einsum('ijj->ij', eris['jc_pp'])
    kc_c = numpy.einsum('ijj->ij', eris['kc_pp'])
    tmp = 6 * kc_c - 2 * jc_c
    h_diag[:,:ncore] += tmp.T
    h_diag[:ncore,:] += tmp
    h_diag[:ncore,:ncore] -= tmp[:,:ncore] * 2

    # part5 and part6 diag
    jc_aa = eris['jc_pp'][:,ncore:nocc,ncore:nocc]
    kc_aa = eris['kc_pp'][:,ncore:nocc,ncore:nocc]
    tmp = numpy.einsum('jik,ik->ij', 3*kc_aa-jc_aa, casdm1)
    h_diag[ncore:nocc,:ncore] -= tmp * 2
    h_diag[:ncore,ncore:nocc] -= tmp.T * 2

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag
    h_diag[:,ncore:nocc] += v_diag.T

    g_orb = casscf.pack_uniq_var(g-g.transpose())
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)
        x_cu = x1[:ncore,ncore:]
        x_av = x1[ncore:nocc,nocc:]
        x_ac = x1[ncore:nocc,:ncore]

        # part7
        x2 = reduce(numpy.dot, (eris['h1e_mo'], x1, dm1))
        # part8, this term destroy Hermitian, symmetrize it.
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{rs} * .5
        x2 -= (numpy.dot(g[:,:nocc], x1[:nocc]) + numpy.dot(g.T, x1)) * .5
        # part2
        x2[:ncore] += numpy.dot(x_cu, vhf_ca[ncore:]) * 2
        # part3
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x_av, vhf_c[nocc:])) \
                        + reduce(numpy.dot, (casdm1, x_ac, vhf_c[:ncore]))
        # part4, part5, part6
        # J3 = eri_popc * pc + eri_cppo * cp
        # K3 = eri_ppco * pc + eri_pcpo * cp
        # J4 = eri_pcpa * pa + eri_appc * ap
        # K4 = eri_ppac * pa + eri_papc * ap
#        vhf3 = numpy.einsum('cpoq,cp->oq', eris['cpop'], x1[:ncore]) * 4 \
#             - numpy.einsum('copq,cq->op', eris['copp'], x1[:ncore]) \
#             - numpy.einsum('cpoq,cq->op', eris['cpop'], x1[:ncore])
#        eri_capp = eris['copp'][:,ncore:nocc,:,:]
#        eri_cpap = eris['cpop'][:,:,ncore:nocc,:]
#        dm4 = numpy.dot(casdm1, x1[ncore:nocc])
#        vhf4 = numpy.einsum('cpuq,uq->cp', eri_cpap, dm4) * 4 \
#             - numpy.einsum('cupq,up->cq', eri_capp, dm4) \
#             - numpy.einsum('cpuq,up->cq', eri_cpap, dm4)
#        x2[ncore:nocc] += numpy.dot(casdm1, vhf3[ncore:nocc])
#        x2[:ncore] += 2 * vhf3[:ncore] + vhf4
        #vhf3c = numpy.einsum('cpia,cp->ia', eris['cPCv'], x1[:ncore])
        if ncore > 0:
            vhf3c = numpy.dot(x1[:ncore].reshape(-1), \
                              eris['cPCv'].reshape(ncore*nmo,-1)).reshape(ncore,-1)
            vhf3a = numpy.einsum('cpuq,cp->uq', eris['cPAp'], x1[:ncore])
            cvap = eris['cPAp'][:,ncore:,:,:]
            dm4 = numpy.dot(casdm1, x1[ncore:nocc])
            vhf4 = numpy.einsum('cpuq,uq->cp', cvap, dm4)
            x2[ncore:nocc] += numpy.dot(casdm1, vhf3a)
            x2[:ncore,ncore:] += 2 * vhf3c + vhf4

        # part1
        x2[ncore:nocc] += numpy.einsum('upvr,vr->up', hdm2, x1[ncore:nocc])

        x2 = x2 - x2.T
        return casscf.pack_uniq_var(x2)
    return g_orb, h_op, h_diag


# update orbital rotation without integral transfromation. No CI gradients were
# calculated in this function.  The gradients was approximately calculated and
# reduced in 3 - 4 iterations of augmented hessian.
def rotate_orb_ah(mol, casscf, mo, fcivec, e_ci, eris, dx=0, verbose=None):
    if verbose is None:
        verbose = casscf.verbose
    log = lib.logger.Logger(casscf.stdout, verbose)

    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nmo = mo.shape[1]

    t2m = (time.clock(), time.time())
    casdm1, casdm2 = fci_direct.make_rdm12(fcivec, ncas, nelecas)
    g_orb0, h_op, h_diag = gen_g_hop(mol, casscf, mo, casdm1, casdm2, eris)
    t3m = log.timer('gen h_op', *t2m)

    precond = lambda x, e: x/(h_diag-(e-casscf.ah_level_shift))
    u = numpy.eye(nmo)

    if isinstance(dx, int):
        x0 = g_orb0
        g_orb = g_orb0
    else:
# h_op is a linear function of dx, so we can start from previous dx as initial
# guess.  But care should be taken to avoid to count the initial guess twice
        x0 = dx
        g_orb = g_orb0 + h_op(dx)
        #if numpy.linalg.norm(g_orb) > numpy.linalg.norm(g_orb0):
        #    return u, dx, g_orb0, 0

    for imic in range(casscf.max_cycle_micro_inner):
        norm_gprev = numpy.linalg.norm(g_orb)
        w, dx1 = aug_hessian.davidson(h_op, g_orb, precond, x0, log, \
                                     tol=casscf.ah_conv_threshold, \
                                     max_cycle=casscf.ah_max_cycle, \
                                     max_stepsize=1.5, \
                                     lindep=casscf.ah_lindep)
        x0 = dx1
        dxmax = numpy.max(abs(dx1))
        if dxmax > casscf.max_orb_stepsize:
            dx1 = dx1 * (casscf.max_orb_stepsize/dxmax)
        dx = dx + dx1
        dr = casscf.unpack_uniq_var(dx1)
        u = numpy.dot(u, expmat(dr))

        g_orb = g_orb0 + h_op(dx)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.debug1('    inner iter %d, |g[o]|=%4.3g', imic, norm_gorb)
# within few steps, g_orb + \sum_i h_op(dx_i) is a good approximation to the
# exact gradients. After few updates, decreasing the approx gradients may
# result in the increase of the real gradient.
        if norm_gorb > norm_gprev \
           or numpy.linalg.norm(dx1) < casscf.conv_threshold_grad \
           or norm_gorb < casscf.conv_threshold_grad:
            break

    t3m = log.timer('aug_hess + gci in %d inner iters' % (imic+1), *t3m)
    return u, dx, g_orb, imic+1

# dc = h_{co} * dr
def hessian_co(mol, casscf, mo, rmat, fcivec, e_ci, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    nocc = ncore + ncas
    mocc = mo[:,:nocc]

    h1eff = numpy.dot(rmat[:,:nocc].T, eris['h1e_mo'][:,:nocc])
    h1eff = h1eff + h1eff.T
    v1 = numpy.einsum('iup,pv->uv', eris['jc_pp'][:,ncore:nocc], rmat[:,ncore:nocc]) \
       + numpy.einsum('uvip,pi->uv', eris['aacp'], rmat[:,:ncore])
    v1 = v1 * 2 \
       - numpy.einsum('iup,pv->uv', eris['kc_pp'][:,ncore:nocc], rmat[:,ncore:nocc]) \
       - numpy.einsum('uivp,pi->uv', eris['acap'], rmat[:,:ncore])
    h1cas = h1eff[ncore:,ncore:] + (v1 + v1.T)

    aaaa = numpy.einsum('tuvp,pw->tuvw', eris['aaap'], rmat[:,ncore:nocc])
    aaaa = aaaa + aaaa.transpose(0,1,3,2)
    aaaa = aaaa + aaaa.transpose(2,3,0,1)
    h2eff = fci_direct.absorb_h1e(h1cas, aaaa, ncas, nelecas) * .5
    hc = fci_direct.contract_2e(h2eff, fcivec, ncas, nelecas).reshape(-1)

    # pure core response
    ecore = h1eff[:ncore,:ncore].trace()*2 \
          + numpy.einsum('ijp,pj->', eris['jc_pp'][:,:ncore], rmat[:,:ncore])*8 \
          - numpy.einsum('ijp,pj->', eris['kc_pp'][:,:ncore], rmat[:,:ncore])*4
    hc += ecore * fcivec.reshape(-1)
    return hc

# dr = h_{oc} * dc
def hessian_oc(mol, casscf, mo, dci, fcivec, eris):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    nocc = ncore + ncas
    mocc = mo[:,:nocc]

    tdm1, tdm2 = fci_direct.trans_rdm12(dci, fcivec, ncas, nelecas)
    tdm1 = (tdm1 + tdm1.T)
    tdm2 = (tdm2 + tdm2.transpose(1,0,3,2))

    inner1 = numpy.dot(dci.flatten(),fcivec.flatten()) * 2

    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2 * inner1 # p^+ q in core, factor due to <c|0> + <0|c>
    dm1[ncore:nocc,ncore:nocc] = tdm1

    vhf_c = numpy.einsum('ipq->pq', eris['jc_pp']) * 2 \
          - numpy.einsum('ipq->pq', eris['kc_pp'])
    vhf_a = numpy.einsum('uvpq,uv->pq', eris['aapp'], tdm1) \
          - numpy.einsum('upqv,uv->pq', eris['appa'], tdm1) * .5

    g2dm = numpy.dot(eris['aaap'].reshape(-1, nmo).T, tdm2.reshape(-1,ncas))

    g = numpy.dot(eris['h1e_mo'], dm1)
    g[:,:ncore] += vhf_a[:,:ncore] * 2
    g[:,:ncore] += vhf_c[:,:ncore] *(2 * inner1)
    g[:,ncore:nocc] += g2dm + numpy.dot(vhf_c[:,ncore:nocc], tdm1)
    return casscf.pack_uniq_var(g - g.transpose())


def kernel(mol, casscf, mo_coeff, tol=1e-7, macro=30, micro=8, \
           ci0=None, verbose=None):
    if verbose is None:
        verbose = casscf.verbose
    log = lib.logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 1-step CASSCF')

    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nmo = mo_coeff.shape[1]

    mo = mo_coeff
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.update_ao2mo(mo)
    e_tot, e_ci, fcivec = casscf.casci(mo, ci0, eris)
    log.info('CASCI E = %.15g', e_tot)
    elast = e_tot
    conv = False
    toloose = casscf.conv_threshold_grad
    totmicro = totinner = 0

    t2m = t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    for imacro in range(macro):
        u, dx, g_orb, ninner = rotate_orb_ah(mol, casscf, mo, fcivec, e_ci, \
                                             eris, 0, verbose=verbose)
        t3m = log.timer('orbital rotation', *t2m)
        for imicro in range(micro):

# approximate newton step, fcivec is not updated during micro iters
            dr = casscf.unpack_uniq_var(dx)
            gci = hessian_co(mol, casscf, mo, dr, fcivec, e_ci, eris)
            t3m = log.timer('ci gradient', *t3m)

# Perturbation updating   gci/(e_ci-hci_diag), gci = H^1 ci^0  is the way
# davidson.dsyev generate new trial vector.  Numerically,
# * Perturbation updating is worse than the simple gradeint.
# * Restorsing the davidson.dsyev hessian from previous FCI solver as the
#   approx CI hessian then solving  (H-E*1)dc = g or aug-hessian or H dc = g
#   has not obvious advantage than simple gradeint.
            dc = gci.reshape(-1)

            #norm_dc = numpy.linalg.norm(dc)
            #if norm_dc > casscf.max_ci_stepsize:
            #    dc = dc * (casscf.max_ci_stepsize/norm_dc)
            dcmax = numpy.max(abs(dc))
            if dcmax > casscf.max_ci_stepsize:
                dc *= casscf.max_ci_stepsize/dcmax
            ci1 = fcivec.reshape(-1) - dc
            ci1 *= 1/numpy.linalg.norm(ci1)

            ovlp_ci = numpy.dot(ci1.reshape(-1), fcivec.reshape(-1))
            norm_gci = numpy.linalg.norm(gci)

            u1, dx, g_orb, nin = rotate_orb_ah(mol, casscf, mo, ci1, e_ci, \
                                               eris, dx, verbose=verbose)
            u = numpy.dot(u, u1)
            ninner += nin
            t3m = log.timer('orbital rotation', *t3m)
            norm_dt = numpy.linalg.norm(u-numpy.eye(nmo))
            norm_gorb = numpy.linalg.norm(g_orb)

            log.debug('micro %d, e_ci = %.12g, |u-1|=%4.3g, |g[o]|=%4.3g, ' \
                      '|g[c]|=%4.3g, max|dc|=%4.3g, <c|c+dc>=%.8g',
                      imicro, e_ci, norm_dt, norm_gorb, \
                      norm_gci, dcmax, ovlp_ci)
            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if (norm_dt < toloose or norm_gorb < toloose or norm_gci < toloose) \
               or dcmax < toloose or 1-ovlp_ci < 1e-8:
                break

        totinner += ninner
        totmicro += imicro+1

        mo = numpy.dot(mo, u)
        eris = None # to avoid using too much memory
        eris = casscf.update_ao2mo(mo)
        t3m = log.timer('update eri', *t3m)

        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris)
        log.info('macro iter %d (%d ah, %d micro), CASSCF E = %.15g, dE = %.8g,',
                 imacro, ninner, imicro+1, e_tot, e_tot-elast)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.info('                        |grad[o]| = %6.5g, |grad[c]| = %6.5g',
                 norm_gorb, norm_gci)
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        #print e_tot, e_tot - elast
        if abs(e_tot - elast) < tol \
           and (norm_gorb < toloose and norm_gci < toloose):
# or abs((elast-e_tot)/e_tot)*1e3 < tol
            conv = True
            break
        else:
            elast = e_tot

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d ah %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d ah %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.log('1-step CASSCF, energy = %.15g', e_tot)
    log.timer('1-step CASSCF', *cput0)
    return e_tot, e_ci, fcivec, mo


class CASSCF(casci.CASCI):
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        casci.CASCI.__init__(self, mol, mf, ncas, nelecas, ncore)
# the max orbital rotation and CI increment, prefer small step size
        self.max_orb_stepsize = .02
        self.max_ci_stepsize = .01
#TODO:self.inner_rotation = False # active-active rotation
        self.max_cycle_macro = 50
# CI vector is approximately updated by gradients. Updating too many times
# seems harmful to convergence.
        self.max_cycle_micro = 3
# num steps to approx orbital rotation without integral transformation.
# Increasing steps do not help converge since the approx gradient might be
# very diff to real gradient after few steps. If the step predicted by AH is
# good enough, it can be set to 1 or 2 steps.
        self.max_cycle_micro_inner = 3
        self.conv_threshold = 1e-7
        self.conv_threshold_grad = 1e-4
        # for augmented hessian
        self.ah_level_shift = 0#1e-2
        self.ah_conv_threshold = 1e-6
        self.ah_max_cycle = 15
        self.ah_lindep = self.ah_conv_threshold**2

        self.e_tot = None
        self.ci = None
        self._hcore = None

    def dump_flags(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** CASSCF flags ********')
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        log.info('CAS (%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas, self.ncas, ncore, nvir)
        log.info('max. macro cycles = %d', self.max_cycle_macro)
        log.info('max. micro cycles = %d', self.max_cycle_micro)
        log.info('conv_threshold = %g, (%g for gradients)', \
                 self.conv_threshold, self.conv_threshold_grad)
        log.info('max_cycle_micro_inner = %d', self.max_cycle_micro_inner)
        log.info('augmented hessian max. cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian conv_threshold = %g', self.ah_conv_threshold)
        log.info('augmented hessian linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian level shift = %d', self.ah_level_shift)
        log.info('max_memory %d MB', self.max_memory)

    def mc1step(self, mo=None, ci0=None, macro=None, micro=None):
        if mo is None:
            mo = self.mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.dump_flags()

        self.e_tot, e_cas, self.ci, self.mo_coeff = \
                kernel(self.mol, self, mo, \
                       tol=self.conv_threshold, macro=macro, micro=micro, \
                       ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def mc2step(self, mo=None, ci0=None, macro=None, micro=None):
        import mc2step
        if mo is None:
            mo = self.mo_coeff
        if macro is None:
            macro = self.max_cycle_macro
        if micro is None:
            micro = self.max_cycle_micro

        self.dump_flags()

        self.e_tot, e_cas, self.ci, self.mo_coeff = \
               mc2step.kernel(self.mol, self, mo, \
                              tol=self.conv_threshold, macro=macro, micro=micro, \
                              ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def casci(self, mo, ci0=None, eris=None):
        if eris is None:
            fcasci = self
        else:
            fcasci = _fake_h_for_fast_casci(self, mo, eris)
        return casci.kernel(self.mol, fcasci, mo, ci0=ci0, verbose=0)

    def pack_uniq_var(self, mat):
        ncore = self.ncore
        nocc = ncore + self.ncas

        v = []
        # active-core
        v.append(mat[ncore:nocc,:ncore].reshape(-1))
        #TODO:if self.inner_rotation:
        #TODO:    # active-active
        #TODO:    v.append(mat[ncore:nocc,ncore:nocc].reshape(-1))
        # virtual-core, virtual-active
        v.append(mat[nocc:,:nocc].reshape(-1))
        return numpy.hstack(v)

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        #TODO:if self.inner_rotation:
        #TODO:    nvir = v.size / nocc - ncas
        #TODO:else:
        #TODO:    nvir = (v.size-ncore*ncas) / nocc
        nvir = (v.size-ncore*ncas) / nocc
        nmo = nocc + nvir

        mat = numpy.zeros((nmo,nmo))
        mat[ncore:nocc,:ncore] = v[:ncas*ncore].reshape(ncas,-1)
        mat[:ncore,ncore:nocc] = -mat[ncore:nocc,:ncore].T
        #TODO:if self.inner_rotation:
        #TODO:    mat[ncore:nocc,ncore:nocc] = v[ncas*ncore:ncas*nocc].reshape(ncas,-1)
        # virtual-core, virtual-active
        mat[nocc:,:nocc] = v[-nvir*nocc:].reshape(nvir,-1)
        mat[:nocc,nocc:] = -mat[nocc:,:nocc].T
        return mat

    # optimize me: mole.moleintor too slow
    def get_hcore(self, mol=None):
        if self._hcore is None:
            self._hcore = self._scf.get_hcore(mol)
        return self._hcore

    def update_ao2mo(self, mo):
        return _ERIS(self, mo)


class _ERIS(dict):
    def __init__(self, casscf, mo):
        mol = casscf.mol
        self.ncore = casscf.ncore
        self.ncas = casscf.ncas
        self.nmo = mo.shape[1]
        nocc = self.ncore + self.ncas
        mocc = mo[:,:nocc]

        self.eri_in_memory = False
        self._erifile = None

        if casscf._scf._eri is not None:
            nao = mo.shape[0]
            npair = nao*(nao+1)/2
            assert(casscf._scf._eri.size == npair*(npair+1)/2)
            eri = ao2mo.incore.general(casscf._scf._eri, (mocc, mo, mo, mo), \
                                       verbose=casscf.verbose)
            self.eri_in_memory = True
        elif nocc*nmo*nmo*nmo*2/1e6 > casscf.max_memory:
            ftmp = tempfile.NamedTemporaryFile()
            #ao2mo.outcore.general(mol, (mocc, mo, mo, mo), ftmp.name, \
            #                      verbose=casscf.verbose)
            ao2mo.direct.general(mol, (mocc, mo, mo, mo), ftmp.name, \
                                 max_memory=self.max_memory, verbose=self.verbose)
            eri = numpy.array(h5py.File(ftmp.name, 'r')['eri_mo'])
        else:
            eri = ao2mo.direct.general_iofree(mol, (mocc, mo, mo, mo), \
                                              verbose=casscf.verbose)
        self._eris = eri
        self['h1e_mo'] = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

# lazy evaluate eris, to reserve enough memory for FCI solver
    def __getitem__(self, key):
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        nmo = self.nmo
        if self.has_key(key) and dict.__getitem__(self, key) is not None:
            return dict.__getitem__(self, key)
        elif key in ('aaaa', 'jc_pp', 'kc_pp', # they are used in casci, gen_g_hop
                     'aacp', 'acap', 'aaap', # they are used in hessian_co
                     'aapp', 'appa'): # they are only used in gen_g_hop
            eri = _mcscf.unpack_eri_tril(self._eris).reshape(nocc,nmo,nmo,nmo)

            self['aaap'] = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
            self['aacp'] = numpy.array(eri[ncore:nocc,ncore:nocc,:ncore,:])
            self['acap'] = numpy.array(eri[ncore:nocc,:ncore,ncore:nocc,:])

            self['jc_pp'] = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
            self['kc_pp'] = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
            self['aaaa'] = numpy.array(self['aaap'][:,:,:,ncore:nocc])
            self['aapp'] = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
            self['appa'] = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
            return dict.__getitem__(self, key)
        elif key in ('cPAp', 'cPCv'):
            # they are only used in gen_g_hop.
            # lazy evaluate them, to reserve the memory for FCI solver
            eri = _mcscf.unpack_eri_tril(self._eris).reshape(nocc,nmo,nmo,nmo)
            capp = eri[:ncore,ncore:nocc,:,:]
            cpap = eri[:ncore,:,ncore:nocc,:]
            ccvp = eri[:ncore,:ncore,ncore:,:]
            cpcv = eri[:ncore,:,:ncore,ncore:]
            cvcp = eri[:ncore,ncore:,:ncore,:]

            self['cPAp'] = cpap * 4 \
                    - capp.transpose(0,3,1,2) \
                    - cpap.transpose(0,3,2,1)
            self['cPCv'] = cpcv * 4 \
                    - ccvp.transpose(0,3,1,2) \
                    - cvcp.transpose(0,3,2,1)
            return dict.__getitem__(self, key)
        else:
            raise KeyError(key)

# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    # vhf for core density matrix
    mo_inv = scipy.linalg.inv(mo)
    vj = numpy.einsum('ipq->pq', eris['jc_pp'])
    vk = numpy.einsum('ipq->pq', eris['kc_pp'])
    vhf = reduce(numpy.dot, (mo_inv.T, vj*2-vk, mo_inv))
    mc.get_veff = lambda *args: vhf

    # eri in CAS space
    idx = numpy.tril_indices(casscf.ncas)
    eri_cas = eris['aaaa'][idx][:,idx[0],idx[1]]
    mc.ao2mo = lambda *args: eri_cas
    return mc


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

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
    emc = kernel(mol, CASSCF(mol, m, 4, 4), m.mo_coeff, verbose=4)[0] + mol.nuclear_repulsion()
    print ehf, emc, emc-ehf
    print emc - -3.22013929407


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASSCF(mol, m, 6, 4)
    mc.verbose = 4
    emc = mc.mc1step()[0] + mol.nuclear_repulsion()
    print ehf, emc, emc-ehf
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print emc - -76.0873923174, emc - -76.0926176464

