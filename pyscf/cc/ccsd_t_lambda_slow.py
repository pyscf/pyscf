#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_lambda

# t2,l2 as ijab

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    saved = ccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    mo_e = mycc._scf.mo_energy
    eia = lib.direct_sum('i-a->ia',mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovvo).transpose(0,1,3,2)
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2)) / d3
    v = numpy.einsum('iajb,kc->ijkabc', eris_ovov, t1) / d3 * .5
    w = p6_(w)
    v = p6_(v)
    rwv = r6_(w*2+v)
    jov = numpy.einsum('jbkc,ijkabc->ia', eris_ovov, r6_(w))
    joovv = numpy.einsum('iabf,ijkabc->kjcf', eris_ovvv, rwv)
    joovv-= numpy.einsum('iajm,ijkabc->mkbc', eris_ovoo, rwv)
    joovv = joovv + joovv.transpose(1,0,3,2)

    saved.jov = jov
    saved.joovv = joovv
    return saved


def update_lambda(mycc, t1, t2, l1, l2, eris=None, saved=None):
    if eris is None: eris = mycc.ao2mo()
    if saved is None: saved = make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    l1, l2 = ccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, saved)

    mo_e = eris.fock.diagonal()
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    l1 += saved.jov/eia * .5

    eijab = lib.direct_sum('ia+jb->ijab', eia, eia)
    l2 += (saved.joovv*(2./3)+saved.joovv.transpose(1,0,2,3)*(1./3)) / eijab
    return l1, l2

def p6_(t):
    t1 = t + t.transpose(0,2,1,3,5,4)
    return t1 + t1.transpose(1,0,2,4,3,5) + t1.transpose(1,2,0,4,5,3)
def r6_(w):
    return (4 * w + w.transpose(0,1,2,4,5,3) + w.transpose(0,1,2,5,3,4)
            - 2 * w.transpose(0,1,2,5,4,3) - 2 * w.transpose(0,1,2,3,5,4)
            - 2 * w.transpose(0,1,2,4,3,5))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()
    #conv, l1, l2 = mcc.solve_lambda()
    #print(numpy.linalg.norm(l1)-0.0132626841292)
    #print(numpy.linalg.norm(l2)-0.212575609057)

    conv, l1, l2 = kernel(mcc, mcc.ao2mo(), t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.013575484203926739)
    print(numpy.linalg.norm(l2)-0.22029981372536928)

