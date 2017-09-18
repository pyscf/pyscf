#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Pipek-Mezey localization
#

import time
import numpy
import scipy.linalg
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.lib import linalg_helper
from pyscf.scf import ciah
from pyscf.lo import orth
from pyscf.lo import boys

def atomic_pops(mol, mo_coeff, method='meta_lowdin'):
    '''kwarg method can be one of mulliken, lowdin, meta_lowdin
    '''
    s = mol.intor_symmetric('int1e_ovlp')
    nmo = mo_coeff.shape[1]
    proj = numpy.empty((mol.natm,nmo,nmo))

    if method.lower() == 'mulliken':
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            csc = reduce(numpy.dot, (mo_coeff[p0:p1].T, s[p0:p1], mo_coeff))
            proj[i] = (csc + csc.T) * .5

    elif method.lower() in ('lowdin', 'meta_lowdin'):
        c = orth.restore_ao_character(mol, 'ANO')
        csc = reduce(lib.dot, (mo_coeff.T, s, orth.orth_ao(mol, method, c, s=s)))
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            proj[i] = numpy.dot(csc[:,p0:p1], csc[:,p0:p1].T)
    else:
        raise KeyError('method = %s' % method)

    return proj


class PipekMezey(boys.Boys):
    def __init__(self, mol, mo_coeff=None):
        boys.Boys.__init__(self, mol, mo_coeff)
        self.pop_method = 'meta_lowdin'
        self.conv_tol = 1e-6
        self._keys = self._keys.union(['pop_method'])

    def dump_flags(self):
        boys.Boys.dump_flags(self)
        logger.info(self, 'pop_method = %s',self.pop_method)

    def gen_g_hop(self, u):
        mo_coeff = lib.dot(self.mo_coeff, u)
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
            norb = x.shape[0]
            hx = lib.dot(x.T, g0.T)
            hx+= numpy.einsum('xip,xi->pi', pop, numpy.einsum('qi,xiq->xi', x, pop)) * 2
            hx-= numpy.einsum('xpp,xip->pi', pop,
                              lib.dot(pop.reshape(-1,norb), x).reshape(-1,norb,norb)) * 2
            hx-= numpy.einsum('xip,xp->pi', pop, numpy.einsum('qp,xpq->xp', x, pop)) * 2
            return -self.pack_uniq_var(hx-hx.T)

        return g, h_op, h_diag

    def get_grad(self, u=None):
        if u is None: u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)
        pop = atomic_pops(self.mol, mo_coeff, self.pop_method)
        g0 = numpy.einsum('xii,xip->pi', pop, pop)
        g = -self.pack_uniq_var(g0-g0.T) * 2
        return g

    def cost_function(self, u=None):
        if u is None: u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)
        pop = atomic_pops(self.mol, mo_coeff, self.pop_method)
        return numpy.einsum('xii,xii->', pop, pop)

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
