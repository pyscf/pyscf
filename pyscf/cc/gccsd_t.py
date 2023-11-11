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
GHF-CCSD(T) with spin-orbital integrals
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import gccsd

# spin-orbital formula
# JCP 98, 8718 (1993); DOI:10.1063/1.464480
def kernel(cc, eris, t1=None, t2=None, verbose=logger.INFO):
    assert (isinstance(eris, gccsd._PhysicistsERIs))
    if t1 is None or t2 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nvir = t1.shape

    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)
    fvo = eris.fock[nocc:,:nocc]
    mo_e = eris.mo_energy
    eijk = lib.direct_sum('i+j+k->ijk', mo_e[:nocc], mo_e[:nocc], mo_e[:nocc])
    eabc = lib.direct_sum('a+b+c->abc', mo_e[nocc:], mo_e[nocc:], mo_e[nocc:])

    t2T = t2.transpose(2,3,0,1)
    t1T = t1.T
    def get_wv(a, b, c):
        w  = numpy.einsum('ejk,ei->ijk', t2T[a,:], bcei[b,c])
        w -= numpy.einsum('im,mjk->ijk', t2T[b,c], majk[:,a])
        v  = numpy.einsum('i,jk->ijk', t1T[a], bcjk[b,c])
        v += numpy.einsum('i,jk->ijk', fvo[a], t2T [b,c])
        v += w
        w = w + w.transpose(2,0,1) + w.transpose(1,2,0)
        return w, v

    et = 0
    for a in range(nvir):
        for b in range(a):
            for c in range(b):
                wabc, vabc = get_wv(a, b, c)
                wcab, vcab = get_wv(c, a, b)
                wbac, vbac = get_wv(b, a, c)

                w = wabc + wcab - wbac
                v = vabc + vcab - vbac
                w /= eijk - eabc[a,b,c]
                et += numpy.einsum('ijk,ijk', w, v.conj())
    et /= 2
    return et


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-1)
    mycc = cc.CCSD(mf).set(conv_tol=1e-11).run()
    et = mycc.ccsd_t()

    mycc = cc.GCCSD(mf.to_ghf()).set(conv_tol=1e-11).run()
    eris = mycc.ao2mo()
    print(kernel(mycc, eris) - et)

    numpy.random.seed(1)
    mf.mo_coeff = numpy.random.random(mf.mo_coeff.shape) - .9
    mycc = cc.GCCSD(mf.to_ghf())
    eris = mycc.ao2mo()
    nocc = 10
    nvir = mol.nao_nr() * 2 - nocc
    t1 = numpy.random.random((nocc,nvir))*.1 - .2
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .2
    t2 = t2 - t2.transpose(1,0,2,3)
    t2 = t2 - t2.transpose(0,1,3,2)
    print(kernel(mycc, eris, t1, t2) - 263713.3945021223)
