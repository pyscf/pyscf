#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd

BLKMIN = 4

'''
CCSD(T)
'''

# t3 as ijkabc

# default max_memory = 2000 MB

# JCP, 94, 442.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2

    nocc, nvir = t1.shape
    nmo = nocc + nvir
    mo_e = cc._scf.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    eiajb = eia.reshape(-1,1) + eia.reshape(-1)
    d3 = eia.reshape(-1,1) + eiajb.reshape(-1)
    d3 = d3.reshape(nocc,nvir,nocc,nvir,nocc,nvir).transpose(0,2,4,1,3,5)

    eris_ovvv = _ccsd.unpack_tril(eris.ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    eris_ovoo = eris.ovoo
    eris_ovov = eris.ovov
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2))
    #r = r6_(w)
    #v = numpy.einsum('iajb,kc->ijkabc', eris_ovov, t1)
    #wvd = p6_(w + .5 * v) / d3
    #et = numpy.einsum('ijkabc,ijkabc', wvd, r) * 2

#    wvd = (w + .5 * v) / d3
#    et =(numpy.einsum('ijkabc,ijkabc', wvd, r)
#       + numpy.einsum('ikjacb,ijkabc', wvd, r)
#       + numpy.einsum('jikbac,ijkabc', wvd, r)
#       + numpy.einsum('jkibca,ijkabc', wvd, r)
#       + numpy.einsum('kijcab,ijkabc', wvd, r)
#       + numpy.einsum('kjicba,ijkabc', wvd, r)) * 2

    r = r6_(p6_(w))
    v = numpy.einsum('iajb,kc->ijkabc', eris_ovov, t1)
    wvd = p6_(w + .5 * v) / d3
    wt = numpy.zeros((nvir,nvir,nvir))
    for i in range(nvir):
        for j in range(i+1):
            for k in range(j+1):
                wt[i,j,k] = 1
                if i == j or j == k:
                    wt[i,j,k] = 1./2
                    if i == k:
                        wt[i,j,k] *= 1./3
    et = numpy.einsum('ijkabc,ijkabc,abc', wvd, r, wt) * 2
    return et

def energy(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    return kernel(cc, eris, t1, t2, max_memory, verbose)

def p6_(t):
    #return (t + t.transpose(1,2,0,4,5,3) +
    #        t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
    #        t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    #t1 = t + t.transpose(1,0,2,4,3,5) + t.transpose(2,0,1,5,3,4)
    #return t1 + t1.transpose(0,2,1,3,5,4)
    t1 = t + t.transpose(0,2,1,3,5,4)
    return t1 + t1.transpose(1,0,2,4,3,5) + t1.transpose(1,2,0,4,5,3)
def r6_(w):
    return (4 * w + w.transpose(0,1,2,4,5,3) + w.transpose(0,1,2,5,3,4)
            - 2 * w.transpose(0,1,2,5,4,3) - 2 * w.transpose(0,1,2,3,5,4)
            - 2 * w.transpose(0,1,2,4,3,5))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc
    from pyscf.cc import ccsd_t_o0

    mol = gto.Mole()
    #mol.verbose = 5
    #mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    print(mcc.ecc)

    e3b = ccsd_t_o0.kernel2(mcc, mcc.ao2mo())
    print(e3b, mcc.ecc+e3b)
    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a, mcc.ecc+e3a)
