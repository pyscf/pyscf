#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import sys
import time
import numpy
from functools import reduce

from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.lo import boys


class EdmistonRuedenberg(boys.Boys):

    def get_jk(self, u):
        mo_coeff = numpy.dot(self.mo_coeff, u)
        nmo = mo_coeff.shape[1]
        dms = [numpy.einsum('i,j->ij', mo_coeff[:,i], mo_coeff[:,i]) for i in range(nmo)]
        vj, vk = hf.get_jk(self.mol, dms, hermi=1)
        vj = numpy.asarray([reduce(numpy.dot, (mo_coeff.T, v, mo_coeff)) for v in vj])
        vk = numpy.asarray([reduce(numpy.dot, (mo_coeff.T, v, mo_coeff)) for v in vk])
        return vj, vk

    def gen_g_hop(self, u):
        vj, vk = self.get_jk(u)

        g0 = numpy.einsum('iip->pi', vj)
        g = -self.pack_uniq_var(g0-g0.T) * 2

        h_diag = numpy.einsum('ipp->pi', vj) * 2
        g_diag = g0.diagonal()
        h_diag-= g_diag + g_diag.reshape(-1,1)
        h_diag+= numpy.einsum('ipp->pi', vk) * 4
        h_diag = -self.pack_uniq_var(h_diag) * 2

        g0 = g0 + g0.T
        def h_op(x):
            x = self.unpack_uniq_var(x)
            hx = numpy.einsum('iq,qp->pi', g0, x)
            hx+= numpy.einsum('qi,iqp->pi', x, vk) * 2
            hx-= numpy.einsum('qp,piq->pi', x, vj) * 2
            hx-= numpy.einsum('qp,piq->pi', x, vk) * 2
            return -self.pack_uniq_var(hx-hx.T)

        return g, h_op, h_diag

    def get_grad(self, u):
        vj, vk = self.get_jk(u)
        g0 = numpy.einsum('iip->pi', vj)
        g = -self.pack_uniq_var(g0-g0.T) * 2
        return g

    def cost_function(self, u):
        vj, vk = self.get_jk(u)
        return numpy.einsum('iii->', vj)

ER = Edmiston = EdmistonRuedenberg

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = '''
         He   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.5   -0.4
      '''
    mol.basis = 'sto-3g'
    mol.build()
    mf = scf.RHF(mol).run()

    mo = ER(mol).kernel(mf.mo_coeff[:,:2], verbose=4)
