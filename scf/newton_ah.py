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
import pyscf.symm
import pyscf.lib.logger as logger

'''
Newton Raphson HF solver with augmented Hessian
'''

def expmat(a):
    return scipy.linalg.expm(a)

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    mol = mf.mol
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if h1e is None: h1e = mf.get_hcore()
    if fock_ao is None:
        vhf0 = mf.get_veff(mol, dm1)
        fock_ao = h1e + vhf0
    else:
        vhf0 = fock_ao - h1e
    fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))

    g = fock[viridx[:,None],occidx] * 2

    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]

    h_diag = (fvv.diagonal().reshape(-1,1)-foo.diagonal()) * 2

    def h_op1(x):
        x = x.reshape(-1,nocc)
        x2 =-numpy.einsum('sq,ps->pq', foo, x) * 2
        x2+= numpy.einsum('pr,rq->pq', fvv, x) * 2
        return x2.reshape(-1)

    def h_opjk(x):
        x = x.reshape(-1,nocc)
        d1 = reduce(numpy.dot, (mo_coeff[:,viridx], x, mo_coeff[:,occidx].T))
        #x2 = reduce(numpy.dot, (mo_coeff[:,viridx].T, mf.get_veff(mol, d1+d1.T),
        #                        mo_coeff[:,occidx])) * 4
        # For DFT
        vhf1 = mf.get_veff(mol, dm0+d1+d1.T, dm_last=dm0, vhf_last=vhf0) - vhf0
        x2 = reduce(numpy.dot, (mo_coeff[:,viridx].T, vhf1,
                                mo_coeff[:,occidx])) * 4
        return x2.reshape(-1)

    return g.reshape(-1), h_op1, h_opjk, h_diag.reshape(-1)


def uniq_var_indices(mo_occ):
    occaidx = mo_occ>0
    occbidx = mo_occ==2
    virbidx = numpy.logical_not(occbidx)
    openidx = numpy.where(mo_occ==1)[0]

    mask = virbidx[:,None]&occaidx
    if len(openidx) > 0:
        mask[openidx[:,None],openidx] = False
    return mask

def pack_uniq_var(x1, mo_occ):
    idx = uniq_var_indices(mo_occ)
    return x1[idx]

def unpack_uniq_var(dx, mo_occ):
    nmo = len(mo_occ)
    idx = uniq_var_indices(mo_occ)

    x1 = numpy.zeros((nmo,nmo))
    x1[idx] = dx
    return x1 - x1.T

def gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    mol = mf.mol
    occidxa = numpy.where(mo_occ>0)[0]
    occidxb = numpy.where(mo_occ==2)[0]
    viridxa = numpy.where(mo_occ==0)[0]
    viridxb = numpy.where(mo_occ<2)[0]
    mask = uniq_var_indices(mo_occ)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if h1e is None: h1e = mf.get_hcore()
    if fock_ao is None:
        vhf0 = mf.get_veff(mol, dm1)
        fock_ao = h1e + vhf0
    else:
        vhf0 = fock_ao - h1e
    focka = reduce(numpy.dot, (mo_coeff.T, fock_ao[0], mo_coeff))
    fockb = reduce(numpy.dot, (mo_coeff.T, fock_ao[1], mo_coeff))

    g = numpy.zeros_like(focka)
    g[viridxa[:,None],occidxa]  = focka[viridxa[:,None],occidxa]
    g[viridxb[:,None],occidxb] += fockb[viridxb[:,None],occidxb]
    g = g[mask]

    h_diag = numpy.zeros_like(focka)
    h_diag[viridxa[:,None],occidxa] -= focka[occidxa,occidxa]
    h_diag[viridxa[:,None],occidxa] += focka[viridxa,viridxa].reshape(-1,1)
    h_diag[viridxb[:,None],occidxb] -= fockb[occidxb,occidxb]
    h_diag[viridxb[:,None],occidxb] += fockb[viridxb,viridxb].reshape(-1,1)
    h_diag = h_diag[mask]

    def h_op1(x):
        x1 = numpy.zeros_like(focka)
        x1[mask] = x
        x1 = x1 - x1.T
        x2 = numpy.zeros_like(focka)

        #: x2[nb:,:na] = numpy.einsum('sp,qs->pq', focka[:na,nb:], x1[:na,:na])
        #: x2[nb:,:na] += numpy.einsum('rq,rp->pq', focka[:na,:na], x1[:na,nb:])
        #: x2[na:,:na] -= numpy.einsum('sp,rp->rs', focka[:na,:na], x1[na:,:na])
        #: x2[na:,:na] -= numpy.einsum('ps,qs->pq', focka[na:], x1[:na]) * 2
        #: x2[nb:na,:nb] += numpy.einsum('qr,pr->pq', focka[:nb], x1[nb:na])
        #: x2[nb:na,:nb] -= numpy.einsum('rq,sq->rs', focka[nb:na], x1[:nb])
        #: x2[nb:,:na] += numpy.einsum('sp,qs->pq', fockb[:nb,nb:], x1[:na,:nb])
        #: x2[nb:,:na] += numpy.einsum('rq,rp->pq', fockb[:nb,:na], x1[:nb,nb:])
        #: x2[nb:,:nb] -= numpy.einsum('sp,rp->rs', fockb[:nb], x1[nb:])
        #: x2[nb:,:nb] -= numpy.einsum('rq,sq->rs', fockb[nb:], x1[:nb]) * 2
        x2[viridxb[:,None],occidxa] = \
                (numpy.einsum('sp,qs->pq', focka[occidxa[:,None],viridxb], x1[occidxa[:,None],occidxa])
                +numpy.einsum('rq,rp->pq', focka[occidxa[:,None],occidxa], x1[occidxa[:,None],viridxb]))
        x2[viridxa[:,None],occidxa] -= \
                (numpy.einsum('sp,rp->rs', focka[occidxa[:,None],occidxa], x1[viridxa[:,None],occidxa])
                +numpy.einsum('ps,qs->pq', focka[viridxa], x1[occidxa]) * 2)
        x2[occidxa[:,None],occidxb] += \
                (numpy.einsum('qr,pr->pq', focka[occidxb], x1[occidxa])
                -numpy.einsum('rq,sq->rs', focka[occidxa], x1[occidxb]))

        x2[viridxb[:,None],occidxa] += \
                (numpy.einsum('sp,qs->pq', fockb[occidxb[:,None],viridxb], x1[occidxa[:,None],occidxb])
                +numpy.einsum('rq,rp->pq', fockb[occidxb[:,None],occidxa], x1[occidxb[:,None],viridxb]))
        x2[viridxb[:,None],occidxb] -= \
                (numpy.einsum('sp,rp->rs', fockb[occidxb], x1[viridxb])
                +numpy.einsum('rq,sq->rs', fockb[viridxb], x1[occidxb]) * 2)
        x2 *= .5
        return x2[mask]

    def h_opjk(x):
        x1 = numpy.zeros_like(focka)
        x1[mask] = x
        x1 = x1 - x1.T
        x2 = numpy.zeros_like(x1)
        d1a = reduce(numpy.dot, (mo_coeff[:,viridxa], x1[viridxa[:,None],occidxa], mo_coeff[:,occidxa].T))
        d1b = reduce(numpy.dot, (mo_coeff[:,viridxb], x1[viridxb[:,None],occidxb], mo_coeff[:,occidxb].T))
        dm1 = numpy.array((dm0[0]+d1a+d1a.T,dm0[1]+d1b+d1b.T))
        vhf1 = mf.get_veff(mol, dm1, dm_last=dm0, vhf_last=vhf0) - vhf0
        x2[viridxa[:,None],occidxa] += reduce(numpy.dot, (mo_coeff[:,viridxa].T, vhf1[0], mo_coeff[:,occidxa]))
        x2[viridxb[:,None],occidxb] += reduce(numpy.dot, (mo_coeff[:,viridxb].T, vhf1[1], mo_coeff[:,occidxb]))
        return x2[mask]

    return g, h_op1, h_opjk, h_diag


def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    mol = mf.mol
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if h1e is None: h1e = mf.get_hcore()
    if fock_ao is None:
        vhf0 = mf.get_veff(mol, dm1)
        fock_ao = h1e + vhf0
    else:
        vhf0 = fock_ao - h1e
    focka = reduce(numpy.dot, (mo_coeff[0].T, fock_ao[0], mo_coeff[0]))
    fockb = reduce(numpy.dot, (mo_coeff[1].T, fock_ao[1], mo_coeff[1]))

    g = numpy.hstack((focka[viridxa[:,None],occidxa].reshape(-1),
                      fockb[viridxb[:,None],occidxb].reshape(-1)))

    h_diaga =(focka[viridxa,viridxa].reshape(-1,1) - focka[occidxa,occidxa])
    h_diagb =(fockb[viridxb,viridxb].reshape(-1,1) - fockb[occidxb,occidxb])
    h_diag = numpy.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))

    def h_op1(x):
        x1a = x[:nvira*nocca].reshape(nvira,nocca)
        x1b = x[nvira*nocca:].reshape(nvirb,noccb)
        x2a = numpy.zeros((nvira,nocca))
        x2b = numpy.zeros((nvirb,noccb))
        x2a -= numpy.einsum('sq,ps->pq', focka[occidxa[:,None],occidxa], x1a)
        x2a += numpy.einsum('rp,rq->pq', focka[viridxa[:,None],viridxa], x1a)
        x2b -= numpy.einsum('sq,ps->pq', fockb[occidxb[:,None],occidxb], x1b)
        x2b += numpy.einsum('rp,rq->pq', fockb[viridxb[:,None],viridxb], x1b)
        return numpy.hstack((x2a.ravel(), x2b.ravel()))

    def h_opjk(x):
        x1a = x[:nvira*nocca].reshape(nvira,nocca)
        x1b = x[nvira*nocca:].reshape(nvirb,noccb)
        d1a = reduce(numpy.dot, (mo_coeff[0][:,viridxa], x1a,
                                 mo_coeff[0][:,occidxa].T))
        d1b = reduce(numpy.dot, (mo_coeff[1][:,viridxb], x1b,
                                 mo_coeff[1][:,occidxb].T))
        dm1 = numpy.array((dm0[0]+d1a+d1a.T,dm0[1]+d1b+d1b.T))
        vhf1 = mf.get_veff(mol, dm1, dm_last=dm0, vhf_last=vhf0) - vhf0
        x2a = reduce(numpy.dot, (mo_coeff[0][:,viridxa].T, vhf1[0],
                                 mo_coeff[0][:,occidxa]))
        x2b = reduce(numpy.dot, (mo_coeff[1][:,viridxb].T, vhf1[1],
                                 mo_coeff[1][:,occidxb]))
        return numpy.hstack((x2a.ravel(), x2b.ravel()))

    return g, h_op1, h_opjk, h_diag


# TODO: check whether high order terms in (g_orb, h_op) affects optimization
# To include high order terms, we can generate mo_coeff every time u matrix
# changed and insert the mo_coeff to g_op, h_op.
# Seems the high order terms do not help optimization?
def rotate_orb_cc(mf, mo_coeff, mo_occ, fock_ao, h1e, verbose=None):
    from pyscf.mcscf import mc1step
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, mf.verbose)

    t2m = (time.clock(), time.time())
    g_orb, h_op1, h_opjk, h_diag = mf.gen_g_hop(mo_coeff, mo_occ, fock_ao, h1e)
    norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|=%4.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t2m)

    def precond(x, e):
        hdiagd = h_diag-(e-mf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return x/hdiagd

    if norm_gorb < 0.01:
        max_cycle = mf.max_cycle_inner-int(numpy.log10(norm_gorb+1e-12))
    else:
        max_cycle = mf.max_cycle_inner

    xcollect = []
    jkcollect = []
    x0 = 0
    x0_guess = g_orb
    ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)
    while True:
        g_orb0 = g_orb
        norm_gprev = norm_gorb
        # increase the AH accuracy when approach convergence
        ah_start_tol = max(min(norm_gorb**2*1e1, mf.ah_start_tol), ah_conv_tol)
        log.debug1('... Set ah_start_tol %g, ah_conv_tol %g',
                   ah_start_tol, ah_conv_tol)
        imic = 0
        wlast = 0
        dx = 0
        u = 1
        jkcount = 0

        g_op = lambda: g_orb
        def h_op(x):
            jk = h_opjk(x)
            if len(xcollect) < mf.ah_guess_space:
                xcollect.append(x)
                jkcollect.append(jk)
            return h_op1(x) + jk
# Divide the hessian into two parts, approx the JK part
# Then clean up the saved JKs to ensure the approximation only get from the
# nearest call
        xsinit = xcollect
        axinit = [h_op1(x)+jkcollect[i] for i,x in enumerate(xcollect)]
        xcollect = []
        jkcollect = []

        for ah_end, ihop, w, dxi, hdxi, residual, seig \
                in mc1step.davidson_cc(h_op, g_op, precond, x0_guess,
                                       xs=xsinit, ax=axinit, verbose=log,
                                       tol=ah_conv_tol, max_cycle=mf.ah_max_cycle,
                                       lindep=mf.ah_lindep):
            if (ah_end or ihop+1 == mf.ah_max_cycle or # make sure to use the last step
                ((abs(w-wlast) < ah_start_tol) and
                 (numpy.linalg.norm(residual)**2 < ah_start_tol) and
                 (ihop >= mf.ah_start_cycle)) or
                (seig < mf.ah_lindep)):
                imic += 1
                dxmax = numpy.max(abs(dxi))
                if dxmax > mf.max_orb_stepsize:
                    scale = mf.max_orb_stepsize / dxmax
                    log.debug1('... scale rotation size %g', scale)
                    dxi *= scale
                    dx = dx + dxi
                    g_orb1 = g_orb + hdxi * scale
                    #g_orb1 = g_orb + h_op1(dxi) + h_opjk(dxi)
                    #jkcount += 1
                else:
                    dx = dx + dxi
                    g_orb1 = g_orb + hdxi  # hdxi not good enough?
                    #g_orb1 = g_orb + h_op1(dxi) + h_opjk(dxi)
                    #jkcount += 1
# Gradually decrease start_tol/conv_tol, so the next step is more accurate
                    ah_start_tol *= mf.ah_decay_rate

                norm_gorb = numpy.linalg.norm(g_orb1)
                norm_dxi = numpy.linalg.norm(dxi)
                log.debug('    inner iter %d(%d)  |g[o]|= %4.3g  |dx|= %4.3g  '
                          'max(|x|)= %4.3g  eig= %4.3g  dw= %4.3g  seig= %4.3g',
                           imic, ihop+1, norm_gorb, norm_dxi, dxmax, w, w-wlast, seig)

                if norm_gorb > norm_gprev * mf.ah_grad_trust_region:
# Do we need force the gradients decaying?
# If in the concave region, how to avoid steping backward (along the negative hessian)?
                    dx -= dxi
                    log.debug('    norm_gorb > nrom_gorb_prev')
                    if numpy.linalg.norm(dx) > 1e-14:
                        break
                else:
                    norm_gprev = norm_gorb
                    g_orb = g_orb1
                    u = mf.update_rotate_matrix(dxi, mo_occ, u)

                if (imic >= max_cycle or norm_gorb < mf.conv_tol_grad*.5):
                    break

            if seig < mf.ah_lindep*1e2 and xcollect:
                xcollect.pop(-1)
                jkcollect.pop(-1)
                log.debug1('... pop xcollect, seig = %g, len(xcollect) = %d',
                           seig, len(xcollect))
            wlast = w

        if numpy.linalg.norm(dx) > 0:
            x0 = x0 + dx
            norm_gorb = numpy.linalg.norm(g_orb)
        else:
            dxi *= .1
            x0 = x0 + dxi
            u = mf.update_rotate_matrix(dxi, mo_occ, u)
            g_orb = g_orb + hdxi * .1
            #g_orb = g_orb + h_op1(dxi) + h_opjk(dxi)
            #jkcount += 1
            norm_gorb = numpy.linalg.norm(g_orb)
            log.debug('orbital rotation step not found, try to guess |g[o]|= %4.3g  |dx|= %4.3g',
                      norm_gorb, numpy.linalg.norm(dxi))

        jkcount += ihop + 2
        log.debug('    tot inner=%d  %d JK  |g[o]|= %4.3g  |u-1|= %4.3g',
                  imic, jkcount, norm_gorb,
                  numpy.linalg.norm(u-numpy.eye(u.shape[1])))
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)
        mo_coeff, mo_occ, fock_ao = (yield u, g_orb0, jkcount)

        g_orb, h_op1, h_opjk, h_diag = mf.gen_g_hop(mo_coeff, mo_occ, fock_ao, h1e)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.debug('    |g|= %4.3g', norm_gorb)
        x0_guess = x0


def kernel(mf, mo_coeff, mo_occ, tol=1e-10, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, verbose)
    mol = mf.mol
    toloose = mf.conv_tol_grad
    scf_conv = False
    hf_energy = mf.hf_energy

    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
# call mf._scf.get_veff, to avoid density_fit module polluting get_veff function
    vhf = mf._scf.get_veff(mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm, 0, None)
    rotaiter = rotate_orb_cc(mf, mo_coeff, mo_occ, fock, h1e, log)
    u, g_orb, jkcount = rotaiter.next()
    jktot = jkcount

    for imacro in range(max_cycle):
        dm_last = dm
        last_hf_e = hf_energy
        norm_gorb = numpy.linalg.norm(g_orb)
        mo_coeff = mf.update_mo_coeff(mo_coeff, u)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, imacro, None)
        mo_energy = mf.get_mo_energy(fock, s1e, mo_coeff, mo_occ)
        mf.get_occ(mo_energy, mo_coeff)
        hf_energy = mf.energy_tot(dm, h1e, vhf)

        log.info('macro= %d  E= %.15g  delta_E= %g  |g|= %g  %d JK',
                 imacro, hf_energy, hf_energy-last_hf_e, norm_gorb,
                 jkcount)

        if (abs((hf_energy-last_hf_e)/hf_energy)*1e2 < tol and
            norm_gorb < toloose):
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

        u, g_orb, jkcount = rotaiter.send((mo_coeff, mo_occ, fock))
        jktot += jkcount

    log.info('macro X = %d  E=%.15g  |g|= %g  total %d JK',
             imacro+1, hf_energy, norm_gorb, jktot)
    mo_energy = mf.get_mo_energy(fock, s1e, mo_coeff, mo_occ)
    mf.get_occ(mo_energy, mo_coeff)
    return scf_conv, hf_energy, mo_energy, mo_coeff, mo_occ


# NOTE: It seems hard to get desired accuracy
# (eg conv_tol=1e-10, conv_tol_grad=1e-5) with density_fit-newton_mf
def newton(mf):
    import pyscf.scf
    from pyscf.mcscf import mc1step_symm
    class Base(mf.__class__):
        def __init__(self):
            self._scf = mf
            self.max_cycle_inner = 6
            self.max_orb_stepsize = .05
            self.conv_tol_grad = numpy.sqrt(mf.conv_tol)*10

            self.ah_start_tol = 1e-2
            self.ah_level_shift = 0
            self.ah_conv_tol = 1e-10
            self.ah_lindep = 1e-14
            self.ah_start_cycle = 2
            self.ah_max_cycle = 30
            self.ah_guess_space = 0
            self.ah_grad_trust_region = 1.5
            self.ah_decay_rate = .5
# * Classic AH can be simulated by setting
#               max_cycle_micro_inner = 1
#               ah_start_tol = 1e-7
#               max_orb_stepsize = 1.5
#               ah_grad_trust_region = 1e6
#               ah_guess_space = 0

            self.__dict__.update(mf.__dict__)
            self._keys = self._keys.union(['max_cycle', 'max_cycle_inner',
                    'max_orb_stepsize', 'conv_tol_grad', 'ah_start_tol',
                    'ah_conv_tol', 'ah_lindep', 'ah_start_cycle',
                    'ah_max_cycle', 'ah_guess_space', 'ah_grad_trust_region'])

        def dump_flags(self):
            logger.info(self, '\n')
            logger.info(self, '******** SCF Newton Raphson flags ********')
            logger.info(self, 'SCF tol = %g', self.conv_tol)
            logger.info(self, 'max. SCF cycles = %d', self.max_cycle)
            logger.info(self, 'direct_scf = %s', self.direct_scf)
            if self.direct_scf:
                logger.info(self, 'direct_scf_tol = %g', self.direct_scf_tol)
            if self.chkfile:
                logger.info(self, 'chkfile to save SCF result = %s', self.chkfile)
            logger.info(self, 'max_cycle_inner = %d',  self.max_cycle_inner)
            logger.info(self, 'max_orb_stepsize = %g', self.max_orb_stepsize)
            logger.info(self, 'conv_tol_grad = %g',    self.conv_tol_grad)
            logger.info(self, 'ah_start_tol = %g',     self.ah_start_tol)
            logger.info(self, 'ah_level_shift = %g',   self.ah_level_shift)
            logger.info(self, 'ah_conv_tol = %g',      self.ah_conv_tol)
            logger.info(self, 'ah_lindep = %g',        self.ah_lindep)
            logger.info(self, 'ah_start_cycle = %d',   self.ah_start_cycle)
            logger.info(self, 'ah_max_cycle = %d',     self.ah_max_cycle)
            logger.info(self, 'ah_guess_space = %d',   self.ah_guess_space)
            logger.info(self, 'ah_grad_trust_region = %g', self.ah_grad_trust_region)
            logger.info(self, 'augmented hessian decay rate = %d', self.ah_decay_rate)

        def get_fock_(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                      diis_start_cycle=None, level_shift_factor=None,
                      damp_factor=None):
            return h1e + vhf

        def kernel(self, mo_coeff=None, mo_occ=None):
            if mo_coeff is None: mo_coeff = self.mo_coeff
            if mo_occ is None: mo_occ = self.mo_occ
            self.dump_flags()
            self.converged, self.hf_energy, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, mo_coeff, mo_occ, tol=self.conv_tol,
                           max_cycle=self.max_cycle,
                           callback=self.callback, verbose=self.verbose)
            return self.hf_energy


    if isinstance(mf, pyscf.scf.hf.ROHF):
        class ROHF(Base):
            def gen_g_hop(self, mo_coeff, mo_occ, fock_ao=None, h1e=None):
                if self.mol.symmetry:
                    self._orbsym = pyscf.symm.label_orb_symm(self.mol,
                            self.mol.irrep_id, self.mol.symm_orb, mo_coeff,
                            s=self.get_ovlp())
                return gen_g_hop_rohf(self, mo_coeff, mo_occ, fock_ao, h1e)

            def update_rotate_matrix(self, dx, mo_occ, u0=1):
                dr = unpack_uniq_var(dx, mo_occ)
                if hasattr(self, '_orbsym'):
                    dr = mc1step_symm._symmetrize(dr, self._orbsym, None)
                return numpy.dot(u0, expmat(dr))

            def update_mo_coeff(self, mo_coeff, u):
                return numpy.dot(mo_coeff, u)

            def get_mo_energy(self, fock, s1e, mo_coeff, mo_occ):
                dm = self.make_rdm1(mo_coeff, mo_occ)
                focka_ao, fockb_ao = fock
                fc = (focka_ao + fockb_ao) * .5
# Projector for core, open-shell, and virtual
                nao = s1e.shape[0]
                pc = numpy.dot(dm[1], s1e)
                po = numpy.dot(dm[0]-dm[1], s1e)
                pv = numpy.eye(nao) - numpy.dot(dm[0], s1e)
                f  = reduce(numpy.dot, (pc.T, fc, pc)) * .5
                f += reduce(numpy.dot, (po.T, fc, po)) * .5
                f += reduce(numpy.dot, (pv.T, fc, pv)) * .5
                f += reduce(numpy.dot, (po.T, fockb_ao, pc))
                f += reduce(numpy.dot, (po.T, focka_ao, pv))
                f += reduce(numpy.dot, (pv.T, fc, pc))
                f = f + f.T
                return self.eig(f, s1e)[0]
                #fc = numpy.dot(fock[0], mo_coeff)
                #mo_energy = numpy.einsum('pk,pk->k', mo_coeff, fc)
                #return mo_energy
        return ROHF()

    elif isinstance(mf, pyscf.scf.uhf.UHF):
        class UHF(Base):
            def gen_g_hop(self, mo_coeff, mo_occ, fock_ao=None, h1e=None):
                if self.mol.symmetry:
                    ovlp_ao = self.get_ovlp()
                    self._orbsym =(pyscf.symm.label_orb_symm(self.mol,
                                            self.mol.irrep_id, self.mol.symm_orb,
                                            mo_coeff[0], s=ovlp_ao),
                                   pyscf.symm.label_orb_symm(self.mol,
                                            self.mol.irrep_id, self.mol.symm_orb,
                                            mo_coeff[1], s=ovlp_ao))
                return gen_g_hop_uhf(self, mo_coeff, mo_occ, fock_ao, h1e)

            def update_rotate_matrix(self, dx, mo_occ, u0=1):
                nmo = len(mo_occ[0])
                occidxa = mo_occ[0]==1
                viridxa = numpy.logical_not(occidxa)
                occidxb = mo_occ[1]==1
                viridxb = numpy.logical_not(occidxb)
                dr = numpy.zeros((2,nmo,nmo))
                idx = numpy.array((viridxa[:,None]&occidxa,
                                   viridxb[:,None]&occidxb))
                dr[idx] = dx
                dr[0] = dr[0] - dr[0].T
                dr[1] = dr[1] - dr[1].T
                if hasattr(self, '_orbsym'):
                    dr = (mc1step_symm._symmetrize(dr[0], self._orbsym[0], None),
                          mc1step_symm._symmetrize(dr[1], self._orbsym[1], None))
                if isinstance(u0, int) and u0 == 1:
                    return numpy.array((expmat(dr[0]), expmat(dr[1])))
                else:
                    return numpy.array((numpy.dot(u0[0], expmat(dr[0])),
                                        numpy.dot(u0[1], expmat(dr[1]))))

            def update_mo_coeff(self, mo_coeff, u):
                return numpy.array(map(numpy.dot, mo_coeff, u))

            def get_mo_energy(self, fock, s1e, mo_coeff, mo_occ):
                return numpy.asarray(self.eig(fock, s1e)[0])
                #fca = numpy.dot(fock[0], mo_coeff[0])
                #fcb = numpy.dot(fock[1], mo_coeff[1])
                #mo_energy =(numpy.einsum('pk,pk->k', mo_coeff[0], fca),
                #            numpy.einsum('pk,pk->k', mo_coeff[1], fcb))
                #return numpy.array(mo_energy)
        return UHF()

    elif isinstance(mf, pyscf.scf.dhf.UHF):
        raise RuntimeError('Not support Dirac-HF')

    else:
        class RHF(Base):
            def gen_g_hop(self, mo_coeff, mo_occ, fock_ao=None, h1e=None):
                if self.mol.symmetry:
                    self._orbsym = pyscf.symm.label_orb_symm(self.mol,
                            self.mol.irrep_id, self.mol.symm_orb, mo_coeff,
                            s=self.get_ovlp())
                return gen_g_hop_rhf(self, mo_coeff, mo_occ, fock_ao, h1e)

            def update_rotate_matrix(self, dx, mo_occ, u0=1):
                nmo = len(mo_occ)
                occidx = mo_occ==2
                viridx = numpy.logical_not(occidx)
                dr = numpy.zeros((nmo,nmo))
                dr[viridx[:,None]&occidx] = dx
                dr = dr - dr.T
                if hasattr(self, '_orbsym'):
                    dr = mc1step_symm._symmetrize(dr, self._orbsym, None)
                return numpy.dot(u0, expmat(dr))

            def update_mo_coeff(self, mo_coeff, u):
                return numpy.dot(mo_coeff, u)

            def get_mo_energy(self, fock, s1e, mo_coeff, mo_occ):
                return self.eig(fock, s1e)[0]
                #fc = numpy.dot(fock, mo_coeff)
                #mo_energy = numpy.einsum('pk,pk->k', mo_coeff, fc)
                #return mo_energy
        return RHF()


def aux_fock_(newton_mf, start_tol=1e-3):
    log = logger.Logger(newton_mf.stdout, newton_mf.verbose)
    newton_mf._last_hf_e = 0
    def get_fock(h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                 diis_start_cycle=None, level_shift_factor=None,
                 damp_factor=None):
        mol = newton_mf.mol
        fock = newton_mf._scf.get_fock_(h1e, s1e, vhf, dm, cycle, adiis)
        hf_energy = newton_mf.energy_tot(dm, h1e, vhf)
        if abs(hf_energy-newton_mf._last_hf_e) > start_tol:
            newton_mf._last_hf_e = hf_energy
            return fock
        else:
            newton_mf._last_hf_e = hf_energy

        mo_energy, mo_coeff = newton_mf._scf.eig(fock, s1e)
        mo_occ = newton_mf._scf.get_occ(mo_energy, mo_coeff)
        dm_new = newton_mf._scf.make_rdm1(mo_coeff, mo_occ)
        vhf = newton_mf._scf.get_veff(mol, dm_new, dm_last=dm, vhf_last=vhf)
        fock = h1e + vhf
        if hasattr(newton_mf, '_ahcc'):
            u, g_orb, jkcount = newton_mf._ahcc.send((mo_coeff, mo_occ, fock))
        else:
            newton_mf._ahcc = rotate_orb_cc(newton_mf, mo_coeff, mo_occ, fock,
                                            h1e, log)
            u, g_orb, jkcount = newton_mf._ahcc.next()
        mo_coeff = newton_mf.update_mo_coeff(mo_coeff, u)
        dm = newton_mf._scf.make_rdm1(mo_coeff, mo_occ)
        vhf = newton_mf._scf.get_veff(mol, dm, dm_last=dm_new, vhf_last=vhf)
        return h1e + vhf

    return get_fock



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0., 1.    , 1.   )],
        ['H', ( 0., 0.5   , 1.   )],
        ['H', ( 1., 0.    ,-1.   )],
    ]

    mol.basis = '6-31g'
    mol.build()

    nmo = mol.nao_nr()
    m = scf.RHF(mol)
    m.scf()
    kernel(newton(m), m.mo_coeff, m.mo_occ, verbose=5)

    m.max_cycle = 1
    m.scf()
    e0 = kernel(newton(m), m.mo_coeff, m.mo_occ, verbose=5)[1]

    mol.basis = '6-31g'
    mol.spin = 2
    mol.build(0, 0)
    m = scf.RHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    e1 = kernel(newton(m), m.mo_coeff, m.mo_occ, max_cycle=50, verbose=5)[1]

    m = scf.UHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    e2 = kernel(newton(m), m.mo_coeff, m.mo_occ, max_cycle=50, verbose=5)[1]

    m = scf.UHF(mol)
    m.get_fock = aux_fock_(newton(m))
    e3 = m.scf()

    m = scf.UHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    nrmf = scf.density_fit(newton(m))
    nrmf.max_cycle = 50
    nrmf.conv_tol = 1e-8
    nrmf.conv_tol_grad = 1e-4
    #nrmf.verbose = 5
    e4 = nrmf.kernel()

    m = scf.density_fit(scf.UHF(mol))
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    nrmf = newton(m)
    nrmf.max_cycle = 50
    e5 = nrmf.kernel()

    print(e0 - -2.93707955256)
    print(e1 - -2.99456398848)
    print(e2 - -2.99663808314)
    print(e3 - -2.99663808314)
    print(e4 - -2.99663808314)
    print(e5 - -2.99634506072)
#    import newton_o2
#    mol.spin = 2
#    mol.build()
#    mf = scf.RHF(mol)
#    mf.scf()
#    nmo = mol.nao_nr()
#    nocc = mol.nelectron // 2
#    nvir = nmo - nocc
#    numpy.random.seed(12)
#    u = numpy.linalg.svd(numpy.random.random((nmo,nmo)))[0]
#    mo_coeff = numpy.dot(mf.mo_coeff, u)
#
#    g0, h_op0, h_diag0 = newton_o2.gen_g_hop(mf, mo_coeff, mf.mo_occ)
#    x = pack_uniq_var(numpy.random.random((nmo,nmo)), mf.mo_occ)
#    xref = h_op0(x)
#
#    mf1 = newton(mf)
#    g1, h_op1, h_op2, h_diag = gen_g_hop_rohf(mf1, mo_coeff, mf.mo_occ)
#    x1 = h_op1(x) + h_op2(x)
#    print(abs(x1-xref).sum())
#    print(abs(g0-g1).sum())
#
#    print(abs(h_diag - h_diag0).sum())
