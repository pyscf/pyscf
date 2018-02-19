#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Lambda equation of GHF-CCSD(T) with spin-orbital integrals

Ref:
JCP, 98, 8718
JCP, 147, 044104
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda
from pyscf.cc import gccsd_lambda


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    imds = gccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    t3c =(numpy.einsum('jkae,bcei->ijkabc', t2, bcei)
        - numpy.einsum('imbc,majk->ijkabc', t2, majk))
    t3c = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
    t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
    t3c /= d3

    t3d = numpy.einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += numpy.einsum('ai,jkbc->ijkabc', eris.fock[nocc:,:nocc], t2)
    t3d = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
    t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
    t3d /= d3

    l1_t = numpy.einsum('ijkabc,jkbc->ia', t3c.conj(), eris.oovv) / eia
    imds.l1_t = l1_t * .25

    m3 = t3c * 2 + t3d
    tmp = numpy.einsum('ijkaef,kbfe->ijab', m3.conj(), eris.ovvv) * .5
    l2_t = tmp - tmp.transpose(0,1,3,2)
    tmp = numpy.einsum('imnabc,mnjc->ijab', m3.conj(), eris.ooov) * .5
    l2_t -= tmp - tmp.transpose(1,0,2,3)
    l2_t += numpy.einsum('kc,ijkabc->ijab', eris.fock[:nocc,nocc:], t3c.conj())
    imds.l2_t = l2_t / lib.direct_sum('ia+jb->ijab', eia, eia)

    return imds


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = gccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 += imds.l1_t
    l2 += imds.l2_t
    return l1, l2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf0 = mf = scf.RHF(mol).run(conv_tol=1)
    mf = scf.addons.convert_to_ghf(mf)
    mycc = cc.GCCSD(mf)
    eris = mycc.ao2mo()
#    mycc.kernel(eris=eris)
#    et = mycc.ccsd_t(eris=eris)
#    conv, l1, l2 = kernel(mycc, eris)
#
#    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
#    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
#    print(lib.finger(l1[0]) --0.01302114832130756  )
#    print(lib.finger(l1[1]) --0.01302114832130756  )
#    print(lib.finger(l2[0]) --0.0066148276888968807)
#    print(lib.finger(l2[1]) --0.013180014994368605 )
#    print(lib.finger(l2[2]) --0.0066148276888968807)
#    print(abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
#    print(abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[0]).max())

    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    mycc0 = cc.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
    l1ref, l2ref = ccsd_t_lambda.update_lambda(mycc0, t1, t2, l1, l2, eris0, imds)
    t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
    t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
    l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(abs(l1[0]-l1ref).max())
    print(abs(l2[1]-l2ref).max())
