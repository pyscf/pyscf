#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Coupled pertubed Hartree-Fock solver
'''

import sys
import time
import numpy
import pyscf.lib
from pyscf.lib import logger


def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose)

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)
    t0 = (time.clock(), time.time())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)
    nocc = e_i.size
    nvir = e_a.size
    nmo = nocc + nvir

    mo1base = h1 * -e_ai

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(h1.shape)
        v *= e_ai
        return v.ravel()
    mo1 = pyscf.lib.krylov(vind_vo, mo1base.ravel(),
                           tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)
    return mo1.reshape(h1.shape), None

# h1 shape is (:,nvir+nocc,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1=None,
                 max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)
    t0 = (time.clock(), time.time())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)
    nocc = e_i.size
    nvir = e_a.size
    nmo = nocc + nvir

    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1.reshape(-1,nmo,nocc)*e_i
    mo_e1 = hs[:,mo_occ>0,:].copy()

    mo1base[:,mo_occ==0] *= -e_ai
    mo1base[:,mo_occ>0] = -s1.reshape(-1,nmo,nocc)[:,mo_occ>0] * .5
    e_ij = e_i.reshape(-1,1) - e_i
    mo_e1 += mo1base[:,mo_occ>0,:] * e_ij

    def vind_vo(mo1):
        v = fvind(mo1.reshape(h1.shape)).reshape(mo1base.shape)
        v[:,mo_occ==0,:] *= e_ai
        v[:,mo_occ>0,:] = 0
        return v.ravel()
    mo1 = pyscf.lib.krylov(vind_vo, mo1base.ravel(),
                           tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1 = mo1.reshape(mo1base.shape)
    log.timer('krylov solver in CPHF', *t0)

    v_mo = fvind(mo1.reshape(h1.shape)).reshape(mo1base.shape)
    mo_e1 -= v_mo[:,mo_occ>0,:]
    mo1[:,mo_occ==0] = mo1base[:,mo_occ==0] - v_mo[:,mo_occ==0]*e_ai

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        return mo1.reshape(h1.shape), mo_e1.reshape(nocc,nocc)

if __name__ == '__main__':
    numpy.random.seed(1)
    nd = 3
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    a = numpy.random.random((nocc*nvir,nocc*nvir))
    a = a + a.T
    def fvind(x):
        v = numpy.dot(a,x[:,nocc:].reshape(-1,nocc*nvir).T)
        v1 = numpy.zeros((nd,nmo,nocc))
        v1[:,nocc:] = v.T.reshape(nd,nvir,nocc)
        return v1
    mo_energy = numpy.sort(numpy.random.random(nmo)) * 10
    mo_occ = numpy.zeros(nmo)
    mo_occ[:nocc] = 2
    e_i = mo_energy[mo_occ>0]
    e_a = mo_energy[mo_occ==0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)
    h1 = numpy.random.random((nd,nmo,nocc))
    h1[:,:nocc,:nocc] = h1[:,:nocc,:nocc] + h1[:,:nocc,:nocc].transpose(0,2,1)
    s1 = numpy.random.random((nd,nmo,nocc))
    s1[:,:nocc,:nocc] = s1[:,:nocc,:nocc] + s1[:,:nocc,:nocc].transpose(0,2,1)

    x = solve(fvind, mo_energy, mo_occ, h1, s1, max_cycle=30)[0]
    print(numpy.linalg.norm(x)-6.272581531366389)
    hs = h1.reshape(-1,nmo,nocc) - s1.reshape(-1,nmo,nocc)*e_i
    print(abs(hs[:,nocc:] + fvind(x)[:,nocc:]+x[:,nocc:]/e_ai).sum())

################
    xref = solve(fvind, mo_energy, mo_occ, h1, s1*0, max_cycle=30)[0][:,mo_occ==0]
    def fvind(x):
        return numpy.dot(a,x.reshape(nd,nocc*nvir).T).T.reshape(nd,nvir,nocc)
    h1 = h1[:,nocc:]
    x0 = numpy.linalg.solve(numpy.diag(1/e_ai.ravel())+a, -h1.reshape(nd,-1).T).T.reshape(nd,nvir,nocc)
    x1 = solve(fvind, mo_energy, mo_occ, h1, max_cycle=30)[0]
    print(abs(x0-x1).sum())
    print(abs(xref-x1).sum())
    print(abs(h1 + fvind(x1)+x1/e_ai).sum())
