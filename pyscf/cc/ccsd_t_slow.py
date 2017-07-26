#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd

'''
CCSD(T)
'''

# t3 as ijkabc

# JCP, 94, 442.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    nocc, nvir = t1.shape
    nmo = nocc + nvir
    e_occ, e_vir = mycc._scf.mo_energy[:nocc], mycc._scf.mo_energy[nocc:]
    eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)

    eris_ovvv = lib.unpack_tril(eris.ovvv.reshape(nocc*nvir,-1))
    eris_vvov = eris_ovvv.reshape(nocc,nvir,nvir,nvir).transpose(1,2,0,3)
    eris_vooo = eris.ovoo.transpose(1,0,2,3)
    eris_vvoo = eris.ovov.transpose(1,3,0,2)
    def get_w(a, b, c):
        w = numpy.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w
    def get_v(a, b, c):
        return numpy.einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])

    et = 0
    for a in range(nvir):
        for b in range(a+1):
            for c in range(b+1):
                d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                if a == c:  # a == b == c
                    d3 *= 6
                elif a == b or b == c:
                    d3 *= 2

                wabc = get_w(a, b, c)
                wacb = get_w(a, c, b)
                wbac = get_w(b, a, c)
                wbca = get_w(b, c, a)
                wcab = get_w(c, a, b)
                wcba = get_w(c, b, a)
                vabc = get_v(a, b, c)
                vacb = get_v(a, c, b)
                vbac = get_v(b, a, c)
                vbca = get_v(b, c, a)
                vcab = get_v(c, a, b)
                vcba = get_v(c, b, a)
                zabc = r3(wabc + .5 * vabc) / d3
                zacb = r3(wacb + .5 * vacb) / d3
                zbac = r3(wbac + .5 * vbac) / d3
                zbca = r3(wbca + .5 * vbca) / d3
                zcab = r3(wcab + .5 * vcab) / d3
                zcba = r3(wcba + .5 * vcba) / d3

                et+= numpy.einsum('ijk,ijk', wabc, zabc)
                et+= numpy.einsum('ikj,ijk', wacb, zabc)
                et+= numpy.einsum('jik,ijk', wbac, zabc)
                et+= numpy.einsum('jki,ijk', wbca, zabc)
                et+= numpy.einsum('kij,ijk', wcab, zabc)
                et+= numpy.einsum('kji,ijk', wcba, zabc)

                et+= numpy.einsum('ijk,ijk', wacb, zacb)
                et+= numpy.einsum('ikj,ijk', wabc, zacb)
                et+= numpy.einsum('jik,ijk', wcab, zacb)
                et+= numpy.einsum('jki,ijk', wcba, zacb)
                et+= numpy.einsum('kij,ijk', wbac, zacb)
                et+= numpy.einsum('kji,ijk', wbca, zacb)

                et+= numpy.einsum('ijk,ijk', wbac, zbac)
                et+= numpy.einsum('ikj,ijk', wbca, zbac)
                et+= numpy.einsum('jik,ijk', wabc, zbac)
                et+= numpy.einsum('jki,ijk', wacb, zbac)
                et+= numpy.einsum('kij,ijk', wcba, zbac)
                et+= numpy.einsum('kji,ijk', wcab, zbac)

                et+= numpy.einsum('ijk,ijk', wbca, zbca)
                et+= numpy.einsum('ikj,ijk', wbac, zbca)
                et+= numpy.einsum('jik,ijk', wcba, zbca)
                et+= numpy.einsum('jki,ijk', wcab, zbca)
                et+= numpy.einsum('kij,ijk', wabc, zbca)
                et+= numpy.einsum('kji,ijk', wacb, zbca)

                et+= numpy.einsum('ijk,ijk', wcab, zcab)
                et+= numpy.einsum('ikj,ijk', wcba, zcab)
                et+= numpy.einsum('jik,ijk', wacb, zcab)
                et+= numpy.einsum('jki,ijk', wabc, zcab)
                et+= numpy.einsum('kij,ijk', wbca, zcab)
                et+= numpy.einsum('kji,ijk', wbac, zcab)

                et+= numpy.einsum('ijk,ijk', wcba, zcba)
                et+= numpy.einsum('ikj,ijk', wcab, zcba)
                et+= numpy.einsum('jik,ijk', wbca, zcba)
                et+= numpy.einsum('jki,ijk', wbac, zcba)
                et+= numpy.einsum('kij,ijk', wacb, zcba)
                et+= numpy.einsum('kji,ijk', wabc, zcba)
    et *= 2
    log.info('CCSD(T) correction = %.15g', et)
    return et

def r3(w):
    return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
            - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
            - 2 * w.transpose(1,0,2))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.M()
    numpy.random.seed(12)
    nocc, nvir = 5, 12
    eris = lambda :None
    eris.ovvv = numpy.random.random((nocc,nvir,nvir*(nvir+1)//2)) * .1
    eris.ovoo = numpy.random.random((nocc,nvir,nocc,nocc)) * .1
    eris.ovov = numpy.random.random((nocc,nvir,nocc,nvir)) * .1
    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    mf = scf.RHF(mol)
    mcc = cc.CCSD(mf)
    mcc._scf.mo_energy = numpy.arange(nocc+nvir)
    print(kernel(mcc, eris, t1, t2) + 8.4953387936460398)

    mol = gto.Mole()
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

    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a - -0.0033300722698513989)

