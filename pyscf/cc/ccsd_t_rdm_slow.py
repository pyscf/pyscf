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
from pyscf.cc.ccsd_t_lambda_slow import p6_, r6_
from pyscf.cc import ccsd_rdm

def gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None):
    doo, dov, dvo, dvv = ccsd_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = ccsd._ERIS(mycc)
    nocc, nvir = t1.shape
    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    mo_e = mycc._scf.mo_energy
    eia = lib.direct_sum('i-a->ia',mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)
    eris_ovoo = eris.ovoo
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2)) / d3
    v = numpy.einsum('iajb,kc->ijkabc', eris.ovov, t1) / d3 * .5
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
    if eris is None: eris = ccsd._ERIS(mycc)

    nocc, nvir = t1.shape
    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    mo_e = mycc._scf.mo_energy
    eia = lib.direct_sum('i-a->ia',mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)
    eris_ovoo = eris.ovoo
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2)) / d3
    v = numpy.einsum('iajb,kc->ijkabc', eris.ovov, t1) / d3 * .5
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


def make_rdm1(mycc, t1, t2, l1, l2, d1=None, eris=None):
    if d1 is None:
        doo, dov, dvo, dvv = gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        doo, dov, dvo, dvv = d1
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[:nocc,nocc:] = dov + dvo.T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].T
    dm1[nocc:,nocc:] = dvv + dvv.T
    for i in range(nocc):
        dm1[i,i] += 2
    return dm1

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, d1=None, d2=None, eris=None):
    if d1 is None:
        doo, dov, dvo, dvv = gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        doo, dov, dvo, dvv = d1
    if d2 is None:
        dovov, dvvvv, doooo, doovv, dovvo, dvovv, dovvv, dooov = \
                gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        dovov, dvvvv, doooo, doovv, dovvo, dvovv, dovvv, dooov = d2

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo))

    dm2[:nocc,nocc:,:nocc,nocc:] = \
            (dovov                   +dovov.transpose(2,3,0,1))
    dm2[nocc:,:nocc,nocc:,:nocc] = \
            (dovov.transpose(1,0,3,2)+dovov.transpose(3,2,1,0))

    dm2[:nocc,:nocc,nocc:,nocc:] = \
            (doovv.transpose(0,1,3,2)+doovv.transpose(1,0,2,3))
    dm2[nocc:,nocc:,:nocc,:nocc] = \
            (doovv.transpose(3,2,0,1)+doovv.transpose(2,3,1,0))
    dm2[:nocc,nocc:,nocc:,:nocc] = \
            (dovvo                   +dovvo.transpose(3,2,1,0))
    dm2[nocc:,:nocc,:nocc,nocc:] = \
            (dovvo.transpose(2,3,0,1)+dovvo.transpose(1,0,3,2))

    dm2[nocc:,nocc:,nocc:,nocc:] = ao2mo.restore(1, dvvvv, nvir)
    dm2[nocc:,nocc:,nocc:,nocc:] *= 4

    doooo = doooo+doooo.transpose(1,0,3,2)
    dm2[:nocc,:nocc,:nocc,:nocc] =(doooo+doooo.transpose(2,3,0,1))

    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0)
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2)

    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2)
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0)

    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[:nocc,nocc:] = dov + dvo.T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].T
    dm1[nocc:,nocc:] = dvv + dvv.T
    for i in range(nocc):
        dm2[i,i,:,:] += dm1 * 2
        dm2[:,:,i,i] += dm1 * 2
        dm2[:,i,i,:] -= dm1
        dm2[i,:,:,i] -= dm1

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2

    return dm2

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _cp(a):
    return numpy.array(a, copy=False, order='C')

def make_theta(t2, out=None):
    return _ccsd.make_0132(t2, t2, 2, -1, out)


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
    h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    e3 =(numpy.einsum('ij,ij->', h1, dm1)
       + numpy.einsum('ijkl,ijkl->', eri_mo, dm2)*.5 + mf.mol.energy_nuc())
    #print e3ref, e3-(mf.e_tot+ecc)

    nocc, nvir = t1.shape
    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    mo_e = mcc._scf.mo_energy
    eia = lib.direct_sum('i-a->ia',mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)
    eris_ovoo = eris.ovoo
    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2)) / d3
    v = numpy.einsum('iajb,kc->ijkabc', eris.ovov, t1) / d3 * .5
    w = ccsd_t.p6_(w)
    v = ccsd_t.p6_(v)
    rw = ccsd_t.r6_(w)
    rwv = ccsd_t.r6_(w*2+v*0)
    dovov = numpy.einsum('kc,ijkabc->iajb', t1, rw)
    dooov =-numpy.einsum('mkbc,ijkabc->jmia', t2, rwv)
    dovvv = numpy.einsum('kjcf,ijkabc->iabf', t2, rwv)
    e3a = numpy.einsum('iajb,iajb', eris.ovov, dovov)
    e3a+= numpy.einsum('iajm,jmia', eris.ovoo, dooov)
    e3a+= numpy.einsum('iabf,iabf', eris_ovvv, dovvv)
    print(e3a)

    w =(numpy.einsum('iabf,kjcf->ijkabc', eris_ovvv, t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo, t2))
    v = numpy.einsum('iajb,kc->ijkabc', eris.ovov, t1)
    wvd = ccsd_t.p6_(w + .5 * v) / d3
    print(numpy.einsum('ijkabc,ijkabc', wvd, ccsd_t.r6_(w)) * 2)
