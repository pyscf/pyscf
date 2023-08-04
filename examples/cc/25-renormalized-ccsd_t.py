#!/usr/bin/env python

'''
Renormalized perturbative triples correction on CCSD (R-CCSD(T)) - slow version

See: Comput. Phys. Commun. 149, 71 (2002); https://doi.org/10.1016/S0010-4655(02)00598-2
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.ccsd_t_slow import r3 

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
    mo_e = eris.mo_energy
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)

    eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
    eris_vooo = numpy.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    eris_vvoo = numpy.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]
    def get_w(a, b, c):
        w = numpy.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w
    def get_v(a, b, c):
        v = numpy.einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= numpy.einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v
    def get_y(a, b, c):
        y = numpy.einsum('i,j,k->ijk', t1T[a], t1T[b], t1T[c]) / 3.
        y+= numpy.einsum('i,jk->ijk', t1T[a], t2T[b,c])
        return y 

    et = 0
    dn = .5
    dn += numpy.einsum('ia,ia', t1, t1)
    tmpt = t2 - .5 * t2.transpose(0,1,3,2)
    tmpc = t2 + numpy.einsum('ia,jb->ijab', t1, t1) 
    dn += numpy.einsum('ijab,ijab', tmpt, tmpc)
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
                yabc = .5 * get_y(a, b, c)
                yacb = .5 * get_y(a, c, b)
                ybac = .5 * get_y(b, a, c)
                ybca = .5 * get_y(b, c, a)
                ycab = .5 * get_y(c, a, b)
                ycba = .5 * get_y(c, b, a)

                et+= numpy.einsum('ijk,ijk', wabc, zabc.conj())
                et+= numpy.einsum('ikj,ijk', wacb, zabc.conj())
                et+= numpy.einsum('jik,ijk', wbac, zabc.conj())
                et+= numpy.einsum('jki,ijk', wbca, zabc.conj())
                et+= numpy.einsum('kij,ijk', wcab, zabc.conj())
                et+= numpy.einsum('kji,ijk', wcba, zabc.conj())

                et+= numpy.einsum('ijk,ijk', wacb, zacb.conj())
                et+= numpy.einsum('ikj,ijk', wabc, zacb.conj())
                et+= numpy.einsum('jik,ijk', wcab, zacb.conj())
                et+= numpy.einsum('jki,ijk', wcba, zacb.conj())
                et+= numpy.einsum('kij,ijk', wbac, zacb.conj())
                et+= numpy.einsum('kji,ijk', wbca, zacb.conj())

                et+= numpy.einsum('ijk,ijk', wbac, zbac.conj())
                et+= numpy.einsum('ikj,ijk', wbca, zbac.conj())
                et+= numpy.einsum('jik,ijk', wabc, zbac.conj())
                et+= numpy.einsum('jki,ijk', wacb, zbac.conj())
                et+= numpy.einsum('kij,ijk', wcba, zbac.conj())
                et+= numpy.einsum('kji,ijk', wcab, zbac.conj())

                et+= numpy.einsum('ijk,ijk', wbca, zbca.conj())
                et+= numpy.einsum('ikj,ijk', wbac, zbca.conj())
                et+= numpy.einsum('jik,ijk', wcba, zbca.conj())
                et+= numpy.einsum('jki,ijk', wcab, zbca.conj())
                et+= numpy.einsum('kij,ijk', wabc, zbca.conj())
                et+= numpy.einsum('kji,ijk', wacb, zbca.conj())

                et+= numpy.einsum('ijk,ijk', wcab, zcab.conj())
                et+= numpy.einsum('ikj,ijk', wcba, zcab.conj())
                et+= numpy.einsum('jik,ijk', wacb, zcab.conj())
                et+= numpy.einsum('jki,ijk', wabc, zcab.conj())
                et+= numpy.einsum('kij,ijk', wbca, zcab.conj())
                et+= numpy.einsum('kji,ijk', wbac, zcab.conj())

                et+= numpy.einsum('ijk,ijk', wcba, zcba.conj())
                et+= numpy.einsum('ikj,ijk', wcab, zcba.conj())
                et+= numpy.einsum('jik,ijk', wbca, zcba.conj())
                et+= numpy.einsum('jki,ijk', wbac, zcba.conj())
                et+= numpy.einsum('kij,ijk', wacb, zcba.conj())
                et+= numpy.einsum('kji,ijk', wabc, zcba.conj())

                dn+= numpy.einsum('ijk,ijk', yabc, zabc.conj())
                dn+= numpy.einsum('ikj,ijk', yacb, zabc.conj())
                dn+= numpy.einsum('jik,ijk', ybac, zabc.conj())
                dn+= numpy.einsum('jki,ijk', ybca, zabc.conj())
                dn+= numpy.einsum('kij,ijk', ycab, zabc.conj())
                dn+= numpy.einsum('kji,ijk', ycba, zabc.conj())

                dn+= numpy.einsum('ijk,ijk', yacb, zacb.conj())
                dn+= numpy.einsum('ikj,ijk', yabc, zacb.conj())
                dn+= numpy.einsum('jik,ijk', ycab, zacb.conj())
                dn+= numpy.einsum('jki,ijk', ycba, zacb.conj())
                dn+= numpy.einsum('kij,ijk', ybac, zacb.conj())
                dn+= numpy.einsum('kji,ijk', ybca, zacb.conj())

                dn+= numpy.einsum('ijk,ijk', ybac, zbac.conj())
                dn+= numpy.einsum('ikj,ijk', ybca, zbac.conj())
                dn+= numpy.einsum('jik,ijk', yabc, zbac.conj())
                dn+= numpy.einsum('jki,ijk', yacb, zbac.conj())
                dn+= numpy.einsum('kij,ijk', ycba, zbac.conj())
                dn+= numpy.einsum('kji,ijk', ycab, zbac.conj())

                dn+= numpy.einsum('ijk,ijk', ybca, zbca.conj())
                dn+= numpy.einsum('ikj,ijk', ybac, zbca.conj())
                dn+= numpy.einsum('jik,ijk', ycba, zbca.conj())
                dn+= numpy.einsum('jki,ijk', ycab, zbca.conj())
                dn+= numpy.einsum('kij,ijk', yabc, zbca.conj())
                dn+= numpy.einsum('kji,ijk', yacb, zbca.conj())

                dn+= numpy.einsum('ijk,ijk', ycab, zcab.conj())
                dn+= numpy.einsum('ikj,ijk', ycba, zcab.conj())
                dn+= numpy.einsum('jik,ijk', yacb, zcab.conj())
                dn+= numpy.einsum('jki,ijk', yabc, zcab.conj())
                dn+= numpy.einsum('kij,ijk', ybca, zcab.conj())
                dn+= numpy.einsum('kji,ijk', ybac, zcab.conj())

                dn+= numpy.einsum('ijk,ijk', ycba, zcba.conj())
                dn+= numpy.einsum('ikj,ijk', ycab, zcba.conj())
                dn+= numpy.einsum('jik,ijk', ybca, zcba.conj())
                dn+= numpy.einsum('jki,ijk', ybac, zcba.conj())
                dn+= numpy.einsum('kij,ijk', yacb, zcba.conj())
                dn+= numpy.einsum('kji,ijk', yabc, zcba.conj())
    et *= 2
    log.info('  CCSD(T) correction = %.15g', et)
    dn *= 2
    log.info('R-CCSD(T) correction = %.15g', et / dn)
    return et / dn


print('> An example in Comput. Phys. Commun. 149, 71 (2002); DOI:10.1016/S0010-4655(02)00598-2')    
import pyscf
from pyscf.data import nist
re = 1.7328 * nist.BOHR
r = re * 3 

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 %f' % r,
    basis = 'dz')

mf = mol.RHF().run(max_cycle=100, verbose=0)

cisolver = pyscf.fci.FCI(mf)
e_fci = cisolver.kernel()[0]
print(' E(FCI)          /  Eh = %.6f' % e_fci)

mycc = mf.CCSD().run(verbose=0)
e_ccsd = mf.e_tot + mycc.e_corr
print(' CCSD error      / mEh = %.3f' % ((e_ccsd - e_fci) * 1e3))

et = mycc.ccsd_t()
e_ccsd_t = e_ccsd + et
print(' CCSD(T) error   / mEh = %.3f' % ((e_ccsd_t - e_fci) * 1e3))
  
eris = mycc.ao2mo(mycc.mo_coeff)
et = kernel(mycc, eris, mycc.t1, mycc.t2) 
e_r_ccsd_t = e_ccsd + et
print(' R-CCSD(T) error / mEh = %.3f' % ((e_r_ccsd_t - e_fci) * 1e3))

