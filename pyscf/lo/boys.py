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

'''
Foster-Boys localization
'''


import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.soscf import ciah
from pyscf.soscf import bfgs
from pyscf.lo import orth, cholesky_mos
from pyscf.lo.stability import stability_newton
from pyscf.tools import mo_mapping
from pyscf import __config__


def kernel(localizer, mo_coeff=None, callback=None, verbose=None):
    if mo_coeff is not None:
        localizer.mo_coeff = numpy.asarray(mo_coeff, order='C')
    if localizer.mo_coeff.shape[1] <= 1:
        return localizer.mo_coeff

    if localizer.verbose >= logger.WARN:
        localizer.check_sanity()
    localizer.dump_flags()

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(localizer, verbose=verbose)

    if localizer.conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(localizer.conv_tol*.1)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = localizer.conv_tol_grad

    if mo_coeff is None:
        if getattr(localizer, 'mol', None) and localizer.mol.natm == 0:
            # For customized Hamiltonian
            u0 = localizer.get_init_guess('random')
        else:
            u0 = localizer.get_init_guess(localizer.init_guess)
    else:
        u0 = localizer.get_init_guess(None)

    cput1 = log.timer('initial guess', *cput0)

    e0 = localizer.cost_function(u0)
    g_orb = localizer.get_grad(u0)
    norm_gorb = numpy.linalg.norm(g_orb)
    log.info('Init f(x)= %.14g  |g|= %g', e0, norm_gorb)

    if localizer.algorithm == 'ciah':
        rotaiter = ciah.rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=log.verbose-1)
    elif localizer.algorithm == 'bfgs':
        rotaiter = bfgs.rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=log.verbose-1,
                                      maximize=localizer.maximize)
    else:
        raise KeyError('Unknown algorithm %s' % (str(localizer.algorithm)))

    u, g_orb, stat = next(rotaiter)
    cput1 = log.timer('initializing CIAH', *cput1)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = e0
    for imacro in range(localizer.max_cycle):
        norm_gorb = numpy.linalg.norm(g_orb)
        u0 = localizer.update_rotation(u0, u)
        e = localizer.cost_function(u0)
        e_last, de = e, e-e_last

        log.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if (norm_gorb < conv_tol_grad and abs(de) < localizer.conv_tol
                and stat.tot_hop < localizer.ah_max_cycle):
            conv = True

        if callable(callback):
            callback(locals())

        if conv:
            break

        u, g_orb, stat = rotaiter.send(u0)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
    log.timer(localizer.__class__.__name__, *cput0)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    localizer.mo_coeff = localizer.sort_orb(u0)
    return localizer.mo_coeff


def dipole_integral(mol, mo_coeff, charge_center=None):
    # The gauge origin has no effects for maximization |<r>|^2
    # Set to charge center for physical significance of <r>
    if getattr(mol, 'pbc_intor', None) is not None:
        raise NotImplementedError('Boys localization for PBC systems is not implemented.')

    if charge_center is None:
        charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                         / mol.atom_charges().sum())
    with mol.with_common_origin(charge_center):
        dip = numpy.asarray([reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff))
                             for x in mol.intor_symmetric('int1e_r', comp=3)])
    return dip

def atomic_init_guess(mol, mo_coeff, kpt=None):
    if getattr(mol, 'pbc_intor', None):
        s = mol.pbc_intor('int1e_ovlp', hermi=1, kpt=kpt)
    else:
        s = mol.intor_symmetric('int1e_ovlp')
    c = orth.orth_ao(mol, s=s)
    mo = reduce(numpy.dot, (c.conj().T, s, mo_coeff))
# Find the AOs which have largest overlap to MOs
    idx = numpy.argsort(numpy.einsum('pi,pi->p', mo.conj(), mo))
    nmo = mo.shape[1]
    idx = sorted(idx[-nmo:])

    # Rotate mo_coeff, make it as close as possible to AOs
    u, w, vh = numpy.linalg.svd(mo[idx])
    return lib.dot(u, vh).conj().T

class OrbitalLocalizer(lib.StreamObject, ciah.CIAHOptimizerMixin):

    conv_tol = getattr(__config__, 'lo_boys_Boys_conv_tol', 1e-6)
    conv_tol_grad = getattr(__config__, 'lo_boys_Boys_conv_tol_grad', None)
    max_cycle = getattr(__config__, 'lo_boys_Boys_max_cycle', 100)
    max_iters = getattr(__config__, 'lo_boys_Boys_max_iters', 20)
    max_stepsize = getattr(__config__, 'lo_boys_Boys_max_stepsize', .05)
    ah_trust_region = getattr(__config__, 'lo_boys_Boys_ah_trust_region', 3)
    ah_start_tol = getattr(__config__, 'lo_boys_Boys_ah_start_tol', 1e9)
    ah_max_cycle = getattr(__config__, 'lo_boys_Boys_ah_max_cycle', 40)
    init_guess = getattr(__config__, 'lo_boys_Boys_init_guess', 'atomic')
    algorithm = getattr(__config__, 'lo_boys_Boys_algorithm', 'ciah')
    maximize = getattr(__config__, 'lo_boys_Boys_maximize', False)

    _keys = {
        'conv_tol', 'conv_tol_grad', 'max_cycle', 'max_iters',
        'max_stepsize', 'ah_trust_region', 'ah_start_tol',
        'ah_max_cycle', 'init_guess', 'algorithm', 'maximize',
        'mol', 'mo_coeff',
    }

    def __init__(self, mol, mo_coeff):
        ciah.CIAHOptimizerMixin.__init__(self, mo_coeff.shape[1])

        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.mo_coeff = mo_coeff

    def rotate_orb(self, u=None):
        if u is None:
            return self.mo_coeff
        else:
            return lib.dot(self.mo_coeff, u)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('conv_tol = %s'       , self.conv_tol       )
        log.info('conv_tol_grad = %s'  , self.conv_tol_grad  )
        log.info('max_cycle = %s'      , self.max_cycle      )
        log.info('max_stepsize = %s'   , self.max_stepsize   )
        log.info('max_iters = %s'      , self.max_iters      )
        log.info('kf_interval = %s'    , self.kf_interval    )
        log.info('kf_trust_region = %s', self.kf_trust_region)
        log.info('ah_start_tol = %s'   , self.ah_start_tol   )
        log.info('ah_start_cycle = %s' , self.ah_start_cycle )
        log.info('ah_level_shift = %s' , self.ah_level_shift )
        log.info('ah_conv_tol = %s'    , self.ah_conv_tol    )
        log.info('ah_lindep = %s'      , self.ah_lindep      )
        log.info('ah_max_cycle = %s'   , self.ah_max_cycle   )
        log.info('ah_trust_region = %s', self.ah_trust_region)
        log.info('init_guess = %s'     , self.init_guess     )
        log.info('algorithm = %s'      , self.algorithm      )

    def get_init_guess(self, key='atomic'):
        '''Generate initial guess for localization.

        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        if isinstance(key, str) and key.lower() == 'atomic':
            u0 = self.init_guess_by_atomic()
        elif isinstance(key, str) and key.lower().startswith('cho'):
            u0 = self.init_guess_by_cholesky()
        else:
            u0 = self.identity_rotation()
        if (isinstance(key, str) and key.lower().startswith('rand')
            or numpy.linalg.norm(self.get_grad(u0)) < 1e-5):
            logger.warn(self, 'Initial orbitals are close to convergence. Adding a '
                        'small perturbation.')
            # Add noise to kick initial guess out of saddle point
            dr = numpy.cos(numpy.arange(self.pdim)) * 1e-3
            u0 = self.extract_rotation(dr)
        return u0

    def init_guess_by_atomic(self):
        return atomic_init_guess(self.mol, self.mo_coeff)

    def init_guess_by_cholesky(self):
        mo_init = cholesky_mos(self.mo_coeff)
        s = self.mol.intor_symmetric('int1e_ovlp')
        return reduce(numpy.dot, (self.mo_coeff.conj().T, s, mo_init))

    def sort_orb(self, u):
        sorted_idx = mo_mapping.mo_1to1map(u)
        return self.rotate_orb(u[:,sorted_idx])

    def stability(self, verbose=None, return_status=False):
        return stability_newton(self, verbose=verbose, return_status=return_status)

    kernel = kernel


@lib.with_doc(OrbitalLocalizer.__doc__)
class OrbitalLocalizerComplex(OrbitalLocalizer, ciah.CIAHOptimizerMixinComplex):

    def __init__(self, mol, mo_coeff):
        ciah.CIAHOptimizerMixinComplex.__init__(self, mo_coeff.shape[1])

        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        # forcing MOs to be complex valued because the rotation is complex-valued
        self.mo_coeff = numpy.asarray(mo_coeff, dtype=numpy.complex128)


class Boys(OrbitalLocalizer):
    r'''
    Base class oflocalization optimizer that maximizes the orbital dipole

    \sum_i | <i| r |i> |^2

    Args:
        mol : Mole object

        mo_coeff : size (N,N) numpy.array
            The orbital space to localize for Boys localization.
            When initializing the localization optimizer ``bopt = Boys(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

    Attributes for Boys class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess. If set
            to 'cholesky', then cholesky orbitals will be used as the initial guess.
            Default is 'atomic'.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''

    def gen_g_hop(self, u=None):
        mo_coeff = self.rotate_orb(u)
        dip = dipole_integral(self.mol, mo_coeff)
        g0 = numpy.einsum('xii,xip->pi', dip, dip)
        g = self.get_grad(u, dip)

        h_diag = numpy.einsum('xii,xpp->pi', dip, dip) * 2
        h_diag-= g0.diagonal() + g0.diagonal().reshape(-1,1)
        h_diag+= numpy.einsum('xip,xip->pi', dip, dip) * 2
        h_diag+= numpy.einsum('xip,xpi->pi', dip, dip) * 2
        h_diag = -self.pack_uniq_var(h_diag) * 4

        #:nmo = mo_coeff.shape[1]
        #:h = numpy.einsum('xjj,xjq,pk->pjqk', dip, dip, numpy.eye(nmo))
        #:h+= numpy.einsum('xqq,xjq,pk->pjqk', dip, dip, numpy.eye(nmo))
        #:h+= numpy.einsum('xjq,xjp,jk->pjqk', dip, dip, numpy.eye(nmo))
        #:h+= numpy.einsum('xjp,xkp,pq->pjqk', dip, dip, numpy.eye(nmo))
        #:h-= numpy.einsum('xjj,xkp,jq->pjqk', dip, dip, numpy.eye(nmo))
        #:h-= numpy.einsum('xpp,xjq,pk->pjqk', dip, dip, numpy.eye(nmo))
        #:h-= numpy.einsum('xjp,xpq,pk->pjqk', dip, dip, numpy.eye(nmo))*2
        #:h = h - h.transpose(0,1,3,2)
        #:h = h - h.transpose(1,0,2,3)
        #:h = h + h.transpose(2,3,0,1)
        #:h *= -.5
        #:idx = numpy.tril_indices(nmo, -1)
        #:h = h[idx][:,idx[0],idx[1]]

        g0 = g0 + g0.conj().T

        def h_op(x):
            x = self.unpack_uniq_var(x)
            norb = x.shape[0]
            #:hx = numpy.einsum('qp,xjj,xjq->pj', x, dip, dip)
            #:hx+= numpy.einsum('qp,xqq,xjq->pj', x, dip, dip)
            #:hx+= numpy.einsum('jk,xkk,xkp->pj', x, dip, dip)
            #:hx+= numpy.einsum('jk,xpp,xkp->pj', x, dip, dip)
            #:hx+= numpy.einsum('qj,xjq,xjp->pj', x, dip, dip)
            #:hx+= numpy.einsum('pk,xjp,xkp->pj', x, dip, dip)
            #:hx-= numpy.einsum('qp,xpp,xjq->pj', x, dip, dip) * 2
            #:hx-= numpy.einsum('qp,xjp,xpq->pj', x, dip, dip) * 2
            #:hx+= numpy.einsum('qj,xjp,xjq->pj', x, dip, dip)
            #:hx+= numpy.einsum('pk,xkp,xjp->pj', x, dip, dip)
            #:hx-= numpy.einsum('jk,xjj,xkp->pj', x, dip, dip) * 2
            #:hx-= numpy.einsum('jk,xkj,xjp->pj', x, dip, dip) * 2
            #:return -self.pack_uniq_var(hx)
            #:hx = numpy.einsum('iq,qp->pi', g0, x)
            hx = lib.dot(x.T, g0.T).conj()
            #:hx+= numpy.einsum('qi,xiq,xip->pi', x, dip, dip) * 2
            hx+= numpy.einsum('xip,xi->pi', dip, numpy.einsum('qi,xiq->xi', x, dip)) * 2
            #:hx-= numpy.einsum('qp,xpp,xiq->pi', x, dip, dip) * 2
            hx-= numpy.einsum('xpp,xip->pi', dip,
                              lib.dot(dip.reshape(-1,norb), x).reshape(3,norb,norb)) * 2
            #:hx-= numpy.einsum('qp,xip,xpq->pi', x, dip, dip) * 2
            hx-= numpy.einsum('xip,xp->pi', dip, numpy.einsum('qp,xpq->xp', x, dip)) * 2
            return -self.pack_uniq_var(hx-hx.conj().T) * 2

        return g, h_op, h_diag

    def get_grad(self, u=None, dip=None):
        if dip is None:
            mo_coeff = self.rotate_orb(u)
            dip = dipole_integral(self.mol, mo_coeff)
        g0 = numpy.einsum('xii,xip->pi', dip, dip)
        g = -self.pack_uniq_var(g0-g0.conj().T) * 4
        return g

    def cost_function(self, u=None):
        mo_coeff = self.rotate_orb(u)
        charge_center = (numpy.einsum('z,zx->x', self.mol.atom_charges(), self.mol.atom_coords())
                         / self.mol.atom_charges().sum())
        dip = dipole_integral(self.mol, mo_coeff, charge_center)
        with self.mol.with_common_origin(charge_center):
            r2 = self.mol.intor_symmetric('int1e_r2')
        r2 = numpy.einsum('pi,pi->', mo_coeff.conj(), lib.dot(r2, mo_coeff)).real
        val = r2 - numpy.einsum('xii,xii->', dip, dip).real
        return val

FB = BF = Boys


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.lo.tools import findiff_grad, findiff_hess

    mol = gto.Mole()
    mol.atom = '''
         O   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.7   -0.2
      '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    log = logger.new_logger(mol, verbose=6)

    mo = mf.mo_coeff[:,:mol.nelectron//2]
    mlo = Boys(mol, mo)

    # Validate gradient and Hessian against finite difference
    g, h_op, hdiag = mlo.gen_g_hop()

    h = numpy.zeros((mlo.pdim,mlo.pdim))
    x0 = mlo.zero_uniq_var()
    for i in range(mlo.pdim):
        x0[i] = 1
        h[:,i] = h_op(x0)
        x0[i] = 0

    def func(x):
        u = mlo.extract_rotation(x)
        f = mlo.cost_function(u)
        if mlo.maximize:
            return -f
        else:
            return f

    def fgrad(x):
        u = mlo.extract_rotation(x)
        return mlo.get_grad(u)

    g_num = findiff_grad(func, x0)
    h_num = findiff_hess(fgrad, x0)
    hdiag_num = numpy.diag(h_num)

    log.info('Grad  error: %.3e', abs(g-g_num).max())
    log.info('Hess  error: %.3e', abs(h-h_num).max())
    log.info('Hdiag error: %.3e', abs(hdiag-hdiag_num).max())

    # localization + stability check using CIAH
    mlo.verbose = 4
    mlo.algorithm = 'ciah'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    # localization + stability check using BFGS
    mlo.algorithm = 'bfgs'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)
