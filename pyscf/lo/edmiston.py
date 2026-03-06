#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Edmiston-Ruedenberg localization
'''

import numpy
from functools import reduce
from pyscf.scf import hf
from pyscf.lo import boys


class EdmistonRuedenberg(boys.OrbitalLocalizer):

    maximize = True

    def get_jk(self, u=None):
        mo_coeff = self.rotate_orb(u)
        nmo = mo_coeff.shape[1]
        dms = [numpy.einsum('i,j->ij', mo_coeff[:,i], mo_coeff[:,i]) for i in range(nmo)]
        vj, vk = hf.get_jk(self.mol, dms, hermi=1)
        vj = numpy.asarray([reduce(numpy.dot, (mo_coeff.T, v, mo_coeff)) for v in vj])
        vk = numpy.asarray([reduce(numpy.dot, (mo_coeff.T, v, mo_coeff)) for v in vk])
        return vj, vk

    def gen_g_hop(self, u=None):
        vj, vk = self.get_jk(u)

        g0 = numpy.einsum('iip->pi', vj)
        g = -self.pack_uniq_var(g0-g0.T) * 4

        h_diag = numpy.einsum('ipp->pi', vj) * 2
        g_diag = g0.diagonal()
        h_diag-= g_diag + g_diag.reshape(-1,1)
        h_diag+= numpy.einsum('ipp->pi', vk) * 4
        h_diag = -self.pack_uniq_var(h_diag) * 4

        g0 = g0 + g0.T

        def h_op(x):
            x = self.unpack_uniq_var(x)
            hx = numpy.einsum('iq,qp->pi', g0, x)
            hx+= numpy.einsum('qi,iqp->pi', x, vk) * 2
            hx-= numpy.einsum('qp,piq->pi', x, vj) * 2
            hx-= numpy.einsum('qp,piq->pi', x, vk) * 2
            return -self.pack_uniq_var(hx-hx.T) * 2

        return g, h_op, h_diag

    def get_grad(self, u=None):
        vj, vk = self.get_jk(u)
        g0 = numpy.einsum('iip->pi', vj)
        g = -self.pack_uniq_var(g0-g0.T) * 4
        return g

    def cost_function(self, u=None):
        vj, vk = self.get_jk(u)
        return numpy.einsum('iii->', vj)

ER = Edmiston = EdmistonRuedenberg

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.lib import logger
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
    mlo = Edmiston(mol, mo)

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
