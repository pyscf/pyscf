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
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm

def gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None):
    doo, dov, dvo, dvv = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = mycc.ao2mo()
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
    wv = w+v
    rw = r6_(w)
    goo =-numpy.einsum('iklabc,jklabc->ij', wv, rw) * .5
    gvv = numpy.einsum('ijkacd,ijkbcd->ab', wv, rw) * .5

    doo += goo
    dvv += gvv
    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None):
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            ccsd_rdm.gamma2_intermediates(mycc, t1, t2, l1, l2)
    if eris is None: eris = mycc.ao2mo()

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
    rw = r6_(w)
    rwv = r6_(w*2+v)
    dovov += numpy.einsum('kc,ijkabc->iajb', t1, rw) * .5
    dooov -= numpy.einsum('mkbc,ijkabc->jmia', t2, rwv)
    # Note "dovvv +=" also changes the value of dvvov
    dovvv += numpy.einsum('kjcf,ijkabc->iabf', t2, rwv)
    dvvov = dovvv.transpose(2,3,0,1)
    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov

def _gamma2_outcore(mycc, t1, t2, l1, l2, eris=None, h5fobj=None):
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
            gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    dvovo = dovov.transpose(3,2,1,0)
    dvvoo = doovv.transpose(2,3,0,1)
    dvoov = dovvo.transpose(2,3,0,1)
    dvovv = dovvv.transpose(1,0,2,3)
    dvooo = dooov.transpose(3,2,1,0)
    return dvovo, dvvvv, doooo, dvvoo, dvoov, dvvov, dvovv, dvooo


def make_rdm1(mycc, t1, t2, l1, l2, d1=None, eris=None):
    if d1 is None:
        d1 = gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2, d1)

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, d1=None, d2=None, eris=None):
    if d1 is None:
        d1 = gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    if d2 is None:
        d2 = gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    return ccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2, d1, d2)


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
    from pyscf.cc import ccsd_t_slow as ccsd_t
    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    from pyscf import ao2mo

    mol = gto.Mole()
    #mol.verbose = 5
    #mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]

    #mol.basis = 'ccpvdz'
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    mcc = ccsd.CCSD(mf)
    mcc.conv_tol = 1e-14
    ecc, t1, t2 = mcc.kernel()
    eris = mcc.ao2mo()
    e3ref = ccsd_t.kernel(mcc, eris, t1, t2)
    l1, l2 = ccsd_t_lambda.kernel(mcc, eris, t1, t2)[1:]
    print(ecc, e3ref)

    eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
    nmo = mf.mo_coeff.shape[1]
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    dm1 = make_rdm1(mcc, t1, t2, l1, l2, eris=eris)
    dm2 = make_rdm2(mcc, t1, t2, l1, l2, eris=eris)
    print(lib.finger(dm1) - 1.2905622485441171)
    print(lib.finger(dm2) - 6.6064384807461831)
    h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    e3 =(numpy.einsum('ij,ij->', h1, dm1)
       + numpy.einsum('ijkl,ijkl->', eri_mo, dm2)*.5 + mf.mol.energy_nuc())
    #print e3ref, e3-(mf.e_tot+ecc)

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    mo_e = mcc._scf.mo_energy
    eia = lib.direct_sum('i-a->ia',mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovvo).transpose(0,1,3,2)
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2)) / d3
    v = numpy.einsum('iajb,kc->ijkabc', eris_ovov, t1) / d3 * .5
    w = p6_(w)
    v = p6_(v)
    rw = r6_(w)
    rwv = r6_(w*2+v*0)
    dovov = numpy.einsum('kc,ijkabc->iajb', t1, rw)
    dooov =-numpy.einsum('mkbc,ijkabc->jmia', t2, rwv)
    dovvv = numpy.einsum('kjcf,ijkabc->iabf', t2, rwv)
    e3a = numpy.einsum('iajb,iajb', eris_ovov, dovov)
    e3a+= numpy.einsum('iajm,jmia', eris_ovoo, dooov)
    e3a+= numpy.einsum('iabf,iabf', eris_ovvv, dovvv)
    print(e3a)

    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2))
    v = numpy.einsum('iajb,kc->ijkabc', eris_ovov, t1)
    wvd = p6_(w + .5 * v) / d3
    print(numpy.einsum('ijkabc,ijkabc', wvd, r6_(w)) * 2)
