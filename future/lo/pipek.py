#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Pipek-Mezey localization
#

import time
import numpy
import scipy.linalg

from pyscf.lib import logger
from pyscf.lib import linalg_helper
from pyscf.scf import iah
from pyscf.lo import orth
from pyscf.lo import boys

def atomic_pops(mol, mo_coeff, method='meta_lowdin'):
    '''kwarg method can be one of mulliken, lowdin, meta_lowdin
    '''
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    nmo = mo_coeff.shape[1]
    proj = numpy.empty((mol.natm,nmo,nmo))

    if method.lower() == 'mulliken':
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            csc = reduce(numpy.dot, (mo_coeff[p0:p1].T, s[p0:p1], mo_coeff))
            proj[i] = (csc + csc.T) * .5

    elif method.lower() in ('lowdin', 'meta_lowdin'):
        csc = reduce(numpy.dot, (mo_coeff.T, s, orth.orth_ao(mol, method, s=s)))
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            proj[i] = numpy.dot(csc[:,p0:p1], csc[:,p0:p1].T)
    else:
        raise KeyError('method = %s' % method)

    return proj


class PipekMezey(iah.IAHOptimizer):
    def __init__(self, mol, mo_coeff=None):
        iah.IAHOptimizer.__init__(self)
        self.mol = mol
        self.conv_tol = 1e-7
        self.conv_tol_grad = None
        self.max_cycle = 100
        self.max_iters = 10
        self.max_stepsize = .05
        self.ah_start_cycle = 3
        self.ah_trust_region = 3

        self.pop_method = 'meta_lowdin'

        self.mo_coeff = mo_coeff

    def gen_g_hop(self, u):
        mo_coeff = numpy.dot(self.mo_coeff, u)
        pop = atomic_pops(self.mol, mo_coeff, self.pop_method)
        g0 = numpy.einsum('xii,xip->pi', pop, pop)
        g = -self.pack_uniq_var(g0-g0.T) * 2

        h_diag = numpy.einsum('xii,xpp->pi', pop, pop) * 2
        g_diag = g0.diagonal()
        h_diag-= g_diag + g_diag.reshape(-1,1)
        h_diag+= numpy.einsum('xip,xip->pi', pop, pop) * 2
        h_diag+= numpy.einsum('xip,xpi->pi', pop, pop) * 2
        h_diag = -self.pack_uniq_var(h_diag) * 2

        g0 = g0 + g0.T
        def h_op(x):
            x = self.unpack_uniq_var(x)
            hx = numpy.einsum('iq,qp->pi', g0, x)
            hx+= numpy.einsum('qi,xiq,xip->pi', x, pop, pop) * 2
            hx-= numpy.einsum('qp,xpp,xiq->pi', x, pop, pop) * 2
            hx-= numpy.einsum('qp,xip,xpq->pi', x, pop, pop) * 2
            return -self.pack_uniq_var(hx-hx.T)

        return g, h_op, h_diag

    def get_grad(self, u):
        mo_coeff = numpy.dot(self.mo_coeff, u)
        pop = atomic_pops(self.mol, mo_coeff, self.pop_method)
        g = numpy.einsum('xii,xip->pi', pop, pop) * 2
        return -self.pack_uniq_var(g)

    def cost_function(self, u):
        mo_coeff = numpy.dot(self.mo_coeff, u)
        pop = atomic_pops(self.mol, mo_coeff, self.pop_method)
        return numpy.einsum('xii,xii->', pop, pop)

    def init_guess(self):
        nmo = self.mo_coeff.shape[1]
        u0 = numpy.eye(nmo)
        if numpy.linalg.norm(self.get_grad(u0)) < 1e-5:
            # Add noise to kick initial guess out of saddle point
            dr = numpy.cos(numpy.arange((nmo-1)*nmo//2)) * 1e-2
            u0 = self.extract_rotation(dr)
        return u0

    kernel = boys.kernel

PM = Pipek = PipekMezey

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = '''
         O   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.5   -0.4
      '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    mo = PM(mol).kernel(mf.mo_coeff[:,5:9], verbose=4)
