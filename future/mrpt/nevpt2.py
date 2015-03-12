#!/usr/bin/env python
#
# Author: Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import _ctypes
import time
import tempfile
from functools import reduce
import numpy
import h5py
import pyscf.lib
from pyscf.lib import logger
from pyscf import fci
from pyscf import mcscf
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

libmc = pyscf.lib.load_library('libmcscf')

NUMERICAL_ZERO = 1e-15
# Ref JCP, 117, 9138

# h1e is the CAS space effective 1e hamiltonian
# h2e is the CAS space 2e integrals in physics notation
# a' -> p
# b' -> q
# c' -> r
# d' -> s

def make_a16(h1e, h2e, dms, civec, norb, nelec, link_index=None):
    dm3 = dms['3']
    dm4 = dms['4']
    if 'f3ca' in dms and 'f3ac' in dms:
        f3ca = dms['f3ca']
        f3ac = dms['f3ac']
    else:
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelecb = nelec//2
        else:
            neleca, nelecb = nelec
        if link_index is None:
            link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
            link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
        else:
            link_indexa, link_indexb = link_index
        eri = h2e.transpose(0,2,1,3)
        f3ca = _contract4pdm('NEVPTkern_cedf_aedf', eri, civec, norb, nelec,
                             (link_indexa,link_indexb))
        f3ac = _contract4pdm('NEVPTkern_aedf_ecdf', eri, civec, norb, nelec,
                             (link_indexa,link_indexb))

    a16 = -numpy.einsum('ib,rpqiac->pqrabc', h1e, dm3)
    a16 += numpy.einsum('ia,rpqbic->pqrabc', h1e, dm3)
    a16 -= numpy.einsum('ci,rpqbai->pqrabc', h1e, dm3)

# qjkiac = acqjki + delta(ja)qcki + delta(ia)qjkc - delta(qc)ajki - delta(kc)qjai
    #:a16 -= numpy.einsum('kbij,rpqjkiac->pqrabc', h2e, dm4)
    a16 -= f3ca.transpose(1,4,0,2,5,3) # c'a'acb'b -> a'b'c'abc
    a16 -= numpy.einsum('kbia,rpqcki->pqrabc', h2e, dm3)
    a16 -= numpy.einsum('kbaj,rpqjkc->pqrabc', h2e, dm3)
    a16 += numpy.einsum('cbij,rpqjai->pqrabc', h2e, dm3)
    fdm2 = numpy.einsum('kbij,rpajki->prab'  , h2e, dm3)
    for i in range(norb):
        a16[:,i,:,:,:,i] += fdm2

    #:a16 += numpy.einsum('ijka,rpqbjcik->pqrabc', h2e, dm4)
    a16 += f3ac.transpose(1,2,0,4,3,5) # c'a'b'bac -> a'b'c'abc

    #:a16 -= numpy.einsum('kcij,rpqbajki->pqrabc', h2e, dm4)
    a16 -= f3ca.transpose(1,2,0,4,3,5) # c'a'b'bac -> a'b'c'abc

    a16 += numpy.einsum('jbij,rpqiac->pqrabc', h2e, dm3)
    a16 -= numpy.einsum('cjka,rpqbjk->pqrabc', h2e, dm3)
    a16 += numpy.einsum('jcij,rpqbai->pqrabc', h2e, dm3)
    return a16

def make_a22(h1e, h2e, dms, civec, norb, nelec, link_index=None):
    dm2 = dms['2']
    dm3 = dms['3']
    dm4 = dms['4']
    if 'f3ca' in dms and 'f3ac' in dms:
        f3ca = dms['f3ca']
        f3ac = dms['f3ac']
    else:
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelecb = nelec//2
        else:
            neleca, nelecb = nelec
        if link_index is None:
            link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
            link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
        else:
            link_indexa, link_indexb = link_index
        eri = h2e.transpose(0,2,1,3)
        f3ca = _contract4pdm('NEVPTkern_cedf_aedf', eri, civec, norb, nelec,
                             (link_indexa,link_indexb))
        f3ac = _contract4pdm('NEVPTkern_aedf_ecdf', eri, civec, norb, nelec,
                             (link_indexa,link_indexb))

    a22 = -numpy.einsum('pb,kipjac->ijkabc', h1e, dm3)
    a22 -= numpy.einsum('pa,kibjpc->ijkabc', h1e, dm3)
    a22 += numpy.einsum('cp,kibjap->ijkabc', h1e, dm3)
    a22 += numpy.einsum('cqra,kibjqr->ijkabc', h2e, dm3)
    a22 -= numpy.einsum('qcpq,kibjap->ijkabc', h2e, dm3)

# qjprac = acqjpr + delta(ja)qcpr + delta(ra)qjpc - delta(qc)ajpr - delta(pc)qjar
    #a22 -= numpy.einsum('pqrb,kiqjprac->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1,5,0,2,4,3) # c'a'acbb'
    fdm2 = numpy.einsum('pqrb,kiqcpr->ikbc', h2e, dm3)
    for i in range(norb):
        a22[:,i,:,i,:,:] -= fdm2
    a22 -= numpy.einsum('pqab,kiqjpc->ijkabc', h2e, dm3)
    a22 += numpy.einsum('pcrb,kiajpr->ijkabc', h2e, dm3)
    a22 += numpy.einsum('cqrb,kiqjar->ijkabc', h2e, dm3)

    #a22 -= numpy.einsum('pqra,kibjqcpr->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1,3,0,4,2,5) # c'a'bb'ac -> a'b'c'abc

    #a22 += numpy.einsum('rcpq,kibjaqrp->ijkabc', h2e, dm4)
    a22 += f3ca.transpose(1,3,0,4,2,5) # c'a'bb'ac -> a'b'c'abc

    a22 += 2.0*numpy.einsum('jb,kiac->ijkabc', h1e, dm2)
    a22 += 2.0*numpy.einsum('pjrb,kiprac->ijkabc', h2e, dm3)
    fdm2  = numpy.einsum('pa,kipc->ikac', h1e, dm2)
    fdm2 -= numpy.einsum('cp,kiap->ikac', h1e, dm2)
    fdm2 -= numpy.einsum('cqra,kiqr->ikac', h2e, dm2)
    fdm2 += numpy.einsum('qcpq,kiap->ikac', h2e, dm2)
    fdm2 += numpy.einsum('pqra,kiqcpr->ikac', h2e, dm3)
    fdm2 -= numpy.einsum('rcpq,kiaqrp->ikac', h2e, dm3)
    for i in range(norb):
        a22[:,i,:,:,i,:] += fdm2 * 2

    return a22


def make_a17(h1e,h2e,dm2,dm3):
    norb = h1e.shape[0]
    h1e = h1e - numpy.einsum('mjjn->mn',h2e)

    a17 = -numpy.einsum('pi,cabi->abcp',h1e,dm2)\
          -numpy.einsum('kpij,cabjki->abcp',h2e,dm3)
    return a17

def make_a19(h1e,h2e,dm1,dm2):
    h1e = h1e - numpy.einsum('mjjn->mn',h2e)

    a19 = -numpy.einsum('pi,ai->ap',h1e,dm1)\
          -numpy.einsum('kpij,ajki->ap',h2e,dm2)
    return a19

def make_a23(h1e,h2e,dm1,dm2,dm3):
    a23 = -numpy.einsum('ip,caib->abcp',h1e,dm2)\
          -numpy.einsum('pijk,cajbik->abcp',h2e,dm3)\
          +2.0*numpy.einsum('bp,ca->abcp',h1e,dm1)\
          +2.0*numpy.einsum('pibk,caik->abcp',h2e,dm2)

    return a23

def make_a25(h1e,h2e,dm1,dm2):

    a25 = -numpy.einsum('pi,ai->ap',h1e,dm1)\
          -numpy.einsum('pijk,jaik->ap',h2e,dm2)\
          +2.0*numpy.einsum('ap->pa',h1e)\
          +2.0*numpy.einsum('piaj,ij->ap',h2e,dm1)

    return a25

def make_hdm3(dm1,dm2,dm3,hdm1,hdm2):
    delta = numpy.eye(dm3.shape[0])
    hdm3 = - numpy.einsum('pb,qrac->pqrabc',delta,hdm2)\
          - numpy.einsum('br,pqac->pqrabc',delta,hdm2)\
          + numpy.einsum('bq,prac->pqrabc',delta,hdm2)*2.0\
          + numpy.einsum('ap,bqcr->pqrabc',delta,dm2)*2.0\
          - numpy.einsum('ap,cr,bq->pqrabc',delta,delta,dm1)*4.0\
          + numpy.einsum('cr,bqap->pqrabc',delta,dm2)*2.0\
          - numpy.einsum('bqapcr->pqrabc',dm3)\
          + numpy.einsum('ar,pc,bq->pqrabc',delta,delta,dm1)*2.0\
          - numpy.einsum('ar,bqcp->pqrabc',delta,dm2)
    return hdm3


def make_hdm2(dm1,dm2):
    delta = numpy.eye(dm2.shape[0])
    dm2 = numpy.einsum('ikjl->ijkl',dm2) -numpy.einsum('jk,il->ijkl',delta,dm1)
    hdm2 = numpy.einsum('klij->ijkl',dm2)\
            + numpy.einsum('il,kj->ijkl',delta,dm1)\
            + numpy.einsum('jk,li->ijkl',delta,dm1)\
            - 2.0*numpy.einsum('ik,lj->ijkl',delta,dm1)\
            - 2.0*numpy.einsum('jl,ki->ijkl',delta,dm1)\
            - 2.0*numpy.einsum('il,jk->ijkl',delta,delta)\
            + 4.0*numpy.einsum('ik,jl->ijkl',delta,delta)

    return hdm2

def make_hdm1(dm1):
    delta = numpy.eye(dm1.shape[0])
    hdm1 = 2.0*delta-dm1.transpose(1,0)
    return hdm1

def make_a3(h1e,h2e,dm1,dm2,hdm1):
    delta = numpy.eye(dm2.shape[0])
    a3 = numpy.einsum('ia,ip->pa',h1e,hdm1)\
            + 2.0*numpy.einsum('ijka,pj,ik->pa',h2e,delta,dm1)\
            - numpy.einsum('ijka,jpik->pa',h2e,dm2)
    return a3

def make_k27(h1e,h2e,dm1,dm2):
    delta = numpy.eye(dm2.shape[0])
    k27 = -numpy.einsum('ai,pi->pa',h1e,dm1)\
         -numpy.einsum('iajk,pkij->pa',h2e,dm2)\
         +numpy.einsum('iaji,pj->pa',h2e,dm1)
    return k27



def make_a7(h1e,h2e,dm1,dm2,dm3):
    #This dm2 and dm3 need to be in the form of norm order
    delta = numpy.eye(dm2.shape[0])
    # a^+_ia^+_ja_ka^l =  E^i_lE^j_k -\delta_{j,l} E^i_k
    rm2 = numpy.einsum('iljk->ijkl',dm2) - numpy.einsum('ik,jl->ijkl',dm1,delta)
    # E^{i,j,k}_{l,m,n} = E^{i,j}_{m,n}E^k_l -\delta_{k,m}E^{i,j}_{l,n}- \delta_{k,n}E^{i,j}_{m,l}
    # = E^i_nE^j_mE^k_l -\delta_{j,n}E^i_mE^k_l -\delta_{k,m}E^{i,j}_{l,n} -\delta_{k,n}E^{i,j}_{m,l}
    rm3 = numpy.einsum('injmkl->ijklmn',dm3)\
        - numpy.einsum('jn,imkl->ijklmn',delta,dm2)\
        - numpy.einsum('km,ijln->ijklmn',delta,rm2)\
        - numpy.einsum('kn,ijml->ijklmn',delta,rm2)

    a7 = -numpy.einsum('bi,pqia->pqab',h1e,rm2)\
         -numpy.einsum('ai,pqbi->pqab',h1e,rm2)\
         -numpy.einsum('kbij,pqkija->pqab',h2e,rm3) \
         -numpy.einsum('kaij,pqkibj->pqab',h2e,rm3) \
         -numpy.einsum('baij,pqij->pqab',h2e,rm2)
    return rm2, a7

def make_a9(h1e,h2e,hdm1,hdm2,hdm3):
    a9 =  numpy.einsum('ib,pqai->pqab',h1e,hdm2)
    a9 += numpy.einsum('ijib,pqaj->pqab',h2e,hdm2)*2.0
    a9 -= numpy.einsum('ijjb,pqai->pqab',h2e,hdm2)
    a9 -= numpy.einsum('ijkb,pkqaij->pqab',h2e,hdm3)
    a9 += numpy.einsum('ia,pqib->pqab',h1e,hdm2)
    a9 -= numpy.einsum('ijja,pqib->pqab',h2e,hdm2)
    a9 -= numpy.einsum('ijba,pqji->pqab',h2e,hdm2)
    a9 += numpy.einsum('ijia,pqjb->pqab',h2e,hdm2)*2.0
    a9 -= numpy.einsum('ijka,pqkjbi->pqab',h2e,hdm3)
    return a9

def make_a12(h1e,h2e,dm1,dm2,dm3):
    a12 = numpy.einsum('ia,qpib->pqab',h1e,dm2)\
        - numpy.einsum('bi,qpai->pqab',h1e,dm2)\
        + numpy.einsum('ijka,qpjbik->pqab',h2e,dm3)\
        - numpy.einsum('kbij,qpajki->pqab',h2e,dm3)\
        - numpy.einsum('bjka,qpjk->pqab',h2e,dm2)\
        + numpy.einsum('jbij,qpai->pqab',h2e,dm2)
    return a12

def make_a13(h1e,h2e,dm1,dm2,dm3):
    delta = numpy.eye(dm3.shape[0])
    a13 = -numpy.einsum('ia,qbip->pqab',h1e,dm2)
    a13 += numpy.einsum('pa,qb->pqab',h1e,dm1)*2.0
    a13 += numpy.einsum('bi,qiap->pqab',h1e,dm2)
    a13 -= numpy.einsum('pa,bi,qi->pqab',delta,h1e,dm1)*2.0
    a13 -= numpy.einsum('ijka,qbjpik->pqab',h2e,dm3)
    a13 += numpy.einsum('kbij,qjapki->pqab',h2e,dm3)
    a13 += numpy.einsum('blma,qmlp->pqab',h2e,dm2)
    a13 += numpy.einsum('kpma,qbkm->pqab',h2e,dm2)*2.0
    a13 -= numpy.einsum('bpma,qm->pqab',h2e,dm1)*2.0
    a13 -= numpy.einsum('lbkl,qkap->pqab',h2e,dm2)
    a13 -= numpy.einsum('ap,mbkl,qlmk->pqab',delta,h2e,dm2)*2.0
    a13 += numpy.einsum('ap,lbkl,qk->pqab',delta,h2e,dm1)*2.0
    return a13


def Sr(mc,orbe, dms, eris=None, verbose=None):
    #The subspace S_r^{(-1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    dm4 = dms['4']

    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_cas,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],mc.ncas,mc.ncas,mc.ncas).transpose(0,2,1,3)
        core_dm = numpy.dot(mo_core,mo_core.T) *2
        core_vhf = mc.get_veff(mc.mol,core_dm)
        h1e_v = reduce(numpy.dot, (mo_virt.T, mc.get_hcore()+core_vhf , mo_cas))
        h1e_v -= numpy.einsum('mbbn->mn',h2e_v)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['aapp'][:,:,nocc:,ncore:nocc].transpose(2,0,3,1)
        h1e_v = eris['h1eff'][nocc:,ncore:nocc] - numpy.einsum('mbbn->mn',h2e_v)


    a16 = make_a16(h1e,h2e,dms, mc.ci, mc.ncas, mc.nelecas)
    a17 = make_a17(h1e,h2e,dm2,dm3)
    a19 = make_a19(h1e,h2e,dm1,dm2)

    ener = numpy.einsum('ipqr,pqrabc,iabc->i',h2e_v,a16,h2e_v)\
        +  numpy.einsum('ipqr,pqra,ia->i',h2e_v,a17,h1e_v)*2.0\
        +  numpy.einsum('ip,pa,ia->i',h1e_v,a19,h1e_v)

    norm = numpy.einsum('ipqr,rpqbac,iabc->i',h2e_v,dm3,h2e_v)\
        +  numpy.einsum('ipqr,rpqa,ia->i',h2e_v,dm2,h1e_v)*2.0\
        +  numpy.einsum('ip,pa,ia->i',h1e_v,dm1,h1e_v)

    mo_ener = orbe[mc.ncore+mc.ncas:]
    norm_t = 0.0
    ener_t = 0.0
    for i in xrange(norm.shape[0]):
      if norm[i] < NUMERICAL_ZERO:
        continue
      else:
        norm_t += norm[i]
        ener_t -= norm[i]/(ener[i]/norm[i] + mo_ener[i])
    return norm_t, ener_t

def Si(mc, orbe, dms, eris=None, verbose=None):
    #Subspace S_i^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    dm4 = dms['4']

    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_cas,mo_core,mo_cas,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mc.ncas,mc.ncore,mc.ncas,mc.ncas).transpose(0,2,1,3)
        core_dm = numpy.dot(mo_core,mo_core.T) *2
        core_vhf = mc.get_veff(mc.mol,core_dm)
        h1e_v = reduce(numpy.dot, (mo_cas.T, mc.get_hcore()+core_vhf , mo_core))
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['aapp'][:,:,ncore:nocc,:ncore].transpose(2,0,3,1)
        h1e_v = eris['h1eff'][ncore:nocc,:ncore]

    a22 = make_a22(h1e,h2e, dms, mc.ci, mc.ncas, mc.nelecas)
    a23 = make_a23(h1e,h2e,dm1,dm2,dm3)
    a25 = make_a25(h1e,h2e,dm1,dm2)
    delta = numpy.eye(mc.ncas)
    dm3_h = numpy.einsum('abef,cd->abcdef',dm2,delta)*2\
            - dm3.transpose(0,1,3,2,4,5)
    dm2_h = numpy.einsum('ab,cd->abcd',dm1,delta)*2\
            - dm2.transpose(0,1,3,2)
    dm1_h = 2*delta- dm1.transpose(1,0)

    ener = numpy.einsum('qpir,pqrabc,baic->i',h2e_v,a22,h2e_v)\
        +  numpy.einsum('qpir,pqra,ai->i',h2e_v,a23,h1e_v)*2.0\
        +  numpy.einsum('pi,pa,ai->i',h1e_v,a25,h1e_v)

    norm = numpy.einsum('qpir,rpqbac,baic->i',h2e_v,dm3_h,h2e_v)\
        +  numpy.einsum('qpir,rpqa,ai->i',h2e_v,dm2_h,h1e_v)*2.0\
        +  numpy.einsum('pi,pa,ai->i',h1e_v,dm1_h,h1e_v)

    return _norm_to_energy(norm, ener, -orbe[:mc.ncore])


def Sijrs(mc,orbe, eris, verbose=None):
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    ncore = mo_core.shape[1]
    nvirt = mo_virt.shape[1]
    nocc = ncore + mc.ncas
    if eris is None:
        erifile = tempfile.NamedTemporaryFile()
        feri = ao2mo.outcore.general(mc.mol, (mo_core,mo_virt,mo_core,mo_virt),
                                     erifile.name, verbose=mc.verbose)
    else:
        feri = eris['cvcv']

    eia = orbe[:ncore,None] - orbe[None,nocc:]
    norm = 0
    e = 0
    with ao2mo.load(feri) as cvcv:
        for i in range(ncore):
            djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
            gi = numpy.array(cvcv[i*nvirt:(i+1)*nvirt], copy=False)
            gi = gi.reshape(nvirt,ncore,nvirt).transpose(1,2,0)
            t2i = (gi.ravel()/djba).reshape(ncore,nvirt,nvirt)
            # 2*ijab-ijba
            theta = gi*2 - gi.transpose(0,2,1)
            norm += numpy.einsum('jab,jab', gi, theta)
            e += numpy.einsum('jab,jab', t2i, theta)
    return norm, e

def Sijr(mc,orbe, dms, eris, verbose=None):
    #Subspace S_ijr^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_cas,mo_core],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],mc.ncore,mc.ncas,mc.ncore).transpose(0,2,1,3)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['apcv'][:,:ncore].transpose(3,0,2,1)
    if 'h1' in dms:
        hdm1 = dms['h1']
    else:
        hdm1 = make_hdm1(dm1)

    a3 = make_a3(h1e,h2e,dm1,dm2,hdm1)
    norm = 2.0*numpy.einsum('rpji,raji,pa->rji',h2e_v,h2e_v,hdm1)\
         - 1.0*numpy.einsum('rpji,raij,pa->rji',h2e_v,h2e_v,hdm1)
    h = 2.0*numpy.einsum('rpji,raji,pa->rji',h2e_v,h2e_v,a3)\
         - 1.0*numpy.einsum('rpji,raij,pa->rji',h2e_v,h2e_v,a3)

    diff = orbe[mc.ncore+mc.ncas:,None,None] - orbe[None,:mc.ncore,None] - orbe[None,None,:mc.ncore]

    return _norm_to_energy(norm, h, diff)

def Srsi(mc,orbe, dms, eris, verbose=None):
    #Subspace S_ijr^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_virt,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],mc.ncore,mo_virt.shape[1],mc.ncas).transpose(0,2,1,3)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['apcv'][:,nocc:].transpose(3,1,2,0)

    k27 = make_k27(h1e,h2e,dm1,dm2)
    norm = 2.0*numpy.einsum('rsip,rsia,pa->rsi',h2e_v,h2e_v,dm1)\
         - 1.0*numpy.einsum('rsip,sria,pa->rsi',h2e_v,h2e_v,dm1)
    h = 2.0*numpy.einsum('rsip,rsia,pa->rsi',h2e_v,h2e_v,k27)\
         - 1.0*numpy.einsum('rsip,sria,pa->rsi',h2e_v,h2e_v,k27)
    diff = orbe[mc.ncore+mc.ncas:,None,None] + orbe[None,mc.ncore+mc.ncas:,None] - orbe[None,None,:mc.ncore]
    return _norm_to_energy(norm, h, diff)

def Srs(mc,orbe, dms, eris=None, verbose=None):
    #Subspace S_rs^{(-2)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    if mo_virt.shape[1] ==0:
        return 0, 0
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_virt,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],mc.ncas,mo_virt.shape[1],mc.ncas).transpose(0,2,1,3)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['appa'][:,nocc:,nocc:].transpose(1,2,0,3)

# check accuracy of make_a7 a7-a7.transpose(2,3,0,1)
    rm2, a7 = make_a7(h1e,h2e,dm1,dm2,dm3)
    norm = 0.5*numpy.einsum('rsqp,rsba,pqba->rs',h2e_v,h2e_v,rm2)
    h = 0.5*numpy.einsum('rsqp,rsba,pqab->rs',h2e_v,h2e_v,a7)
    diff = orbe[mc.ncore+mc.ncas:,None] + orbe[None,mc.ncore+mc.ncas:]
    return _norm_to_energy(norm, h, diff)

def Sij(mc,orbe, dms, eris, verbose=None):
    #Subspace S_ij^{(-2)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    if mo_core.size ==0 :
        return 0.0
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_cas,mo_core,mo_cas,mo_core],compact=False)
        h2e_v = h2e_v.reshape(mc.ncas,mc.ncore,mc.ncas,mc.ncore).transpose(0,2,1,3)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['appa'][:,:ncore,:ncore].transpose(0,3,1,2)

    if 'h1' in dms:
        hdm1 = dms['h1']
    else:
        hdm1 = make_hdm1(dm1)
    if 'h2' in dms:
        hdm2 = dms['h2']
    else:
        hdm2 = make_hdm2(dm1,dm2)
    if 'h3' in dms:
        hdm3 = dms['h3']
    else:
        hdm3 = make_hdm3(dm1,dm2,dm3,hdm1,hdm2)

# check accuracy of make_a9 a9-a9.transpose(2,3,0,1)
    a9 = make_a9(h1e,h2e,hdm1,hdm2,hdm3)
    norm = 0.5*numpy.einsum('qpij,baij,pqab->ij',h2e_v,h2e_v,hdm2)
    h = 0.5*numpy.einsum('qpij,baij,pqab->ij',h2e_v,h2e_v,a9)
    diff = orbe[:mc.ncore,None] + orbe[None,:mc.ncore]
    return _norm_to_energy(norm, h, -diff)


def Sir(mc,orbe, dms, eris, verbose=None):
    #Subspace S_il^{(0)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), mc.ncas).transpose(0,2,1,3)
        h2e_v1 = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_cas,mo_cas],compact=False)
        h2e_v1 = h2e_v1.reshape(mo_virt.shape[1],mc.ncore,mc.ncas,mc.ncas).transpose(0,2,1,3)
        h2e_v2 = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_cas,mo_core],compact=False)
        h2e_v2 = h2e_v2.reshape(mo_virt.shape[1],mc.ncas,mc.ncas,mc.ncore).transpose(0,2,1,3)
        core_dm = numpy.dot(mo_core,mo_core.T)*2
        corevhf = mc.get_veff(mc.mol, core_dm)
    else:
        ncore = mc.ncore
        nocc = mc.ncore + mc.ncas
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['aapp'][:,:,ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v1 = eris['aapp'][:,:,nocc:,:ncore].transpose(2,0,3,1)
        h2e_v2 = eris['appa'][:,nocc:,:ncore].transpose(1,3,0,2)
        h1e_v = eris['h1eff'][nocc:,:ncore]

    norm = numpy.einsum('rpiq,raib,qpab->ir',h2e_v1,h2e_v1,dm2)*2.0\
         - numpy.einsum('rpiq,rabi,qpab->ir',h2e_v1,h2e_v2,dm2)\
         - numpy.einsum('rpqi,raib,qpab->ir',h2e_v2,h2e_v1,dm2)\
         + numpy.einsum('raqi,rabi,qb->ir',h2e_v2,h2e_v2,dm1)*2.0\
         - numpy.einsum('rpqi,rabi,qbap->ir',h2e_v2,h2e_v2,dm2)\
         + numpy.einsum('rpqi,raai,qp->ir',h2e_v2,h2e_v2,dm1)\
         + numpy.einsum('rpiq,ri,qp->ir',h2e_v1,h1e_v,dm1)*4.0\
         - numpy.einsum('rpqi,ri,qp->ir',h2e_v2,h1e_v,dm1)*2.0\
         + numpy.einsum('ri,ri->ir',h1e_v,h1e_v)*2.0

    a12 = make_a12(h1e,h2e,dm1,dm2,dm3)
    a13 = make_a13(h1e,h2e,dm1,dm2,dm3)

    h = numpy.einsum('rpiq,raib,pqab->ir',h2e_v1,h2e_v1,a12)*2.0\
         - numpy.einsum('rpiq,rabi,pqab->ir',h2e_v1,h2e_v2,a12)\
         - numpy.einsum('rpqi,raib,pqab->ir',h2e_v2,h2e_v1,a12)\
         + numpy.einsum('rpqi,rabi,pqab->ir',h2e_v2,h2e_v2,a13)
    diff = orbe[:mc.ncore,None] - orbe[None,mc.ncore+mc.ncas:]
    return _norm_to_energy(norm, h, -diff)



def kernel(mc, *args, **kwargs):
    return sc_nevpt(mc, *args, **kwargs)

def sc_nevpt(mc, verbose=logger.NOTE):
    '''Strongly contracted NEVPT2'''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mc.stdout, mc.verbose)

    time0 = (time.clock(), time.time())
    #dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
    #                                         mc.ci, mc.ci, mc.ncas, mc.nelecas)
    dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf',
                                       mc.ci, mc.ci, mc.ncas, mc.nelecas)
    dm4 = None

    #hdm1 = make_hdm1(dm1)
    #hdm2 = make_hdm2(dm1,dm2)
    #hdm3 = make_hdm3(dm1,dm2,dm3,hdm1,hdm2)

    dms = {'1': dm1, '2': dm2, '3': dm3, '4': dm4,
           #'h1': hdm1, 'h2': hdm2, 'h3': hdm3
          }
    time1 = log.timer('3pdm, 4pdm', *time0)

    eris = _ERIS(mc, mc.mo_coeff)
    time1 = log.timer('integral transformation', *time1)

    link_indexa = fci.cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[0])
    link_indexb = fci.cistring.gen_linkstr_index(range(mc.ncas), mc.nelecas[1])
    nocc = mc.ncore + mc.ncas
    aaaa = eris['aapp'][:,:,mc.ncore:nocc,mc.ncore:nocc].copy()
    f3ca = _contract4pdm('NEVPTkern_cedf_aedf', aaaa, mc.ci, mc.ncas,
                         mc.nelecas, (link_indexa,link_indexb))
    f3ac = _contract4pdm('NEVPTkern_aedf_ecdf', aaaa, mc.ci, mc.ncas,
                         mc.nelecas, (link_indexa,link_indexb))
    dms['f3ca'] = f3ca
    dms['f3ac'] = f3ac
    time1 = log.timer('eri-4pdm contraction', *time1)

    fock =(eris['h1eff']
         + numpy.einsum('ij,ijpq->pq', dm1, eris['aapp'])
         - numpy.einsum('ij,ipqj->pq', dm1, eris['appa']) * .5)
    orbe = fock.diagonal()

    norm_Sr   , e_Sr    = Sr(mc,orbe, dms, eris)
    logger.note(mc, "Sr    (-1)', Norm = %.14f  E = %.14f", norm_Sr  , e_Sr  )
    time1 = log.timer("space Sr (-1)'", *time1)
    norm_Si   , e_Si    = Si(mc,orbe, dms, eris)
    logger.note(mc, "Si    (+1)', Norm = %.14f  E = %.14f", norm_Si  , e_Si  )
    time1 = log.timer("space Si (+1)'", *time1)
    norm_Sijrs, e_Sijrs = Sijrs(mc,orbe, eris)
    logger.note(mc, "Sijrs (0)  , Norm = %.14f  E = %.14f", norm_Sijrs,e_Sijrs)
    time1 = log.timer('space Sijrs (0)', *time1)
    norm_Sijr , e_Sijr  = Sijr(mc,orbe, dms, eris)
    logger.note(mc, "Sijr  (+1) , Norm = %.14f  E = %.14f", norm_Sijr, e_Sijr)
    time1 = log.timer('space Sijr (+1)', *time1)
    norm_Srsi , e_Srsi  = Srsi(mc,orbe, dms, eris)
    logger.note(mc, "Srsi  (-1) , Norm = %.14f  E = %.14f", norm_Srsi, e_Srsi)
    time1 = log.timer('space Srsi (-1)', *time1)
    norm_Srs  , e_Srs   = Srs(mc,orbe, dms, eris)
    logger.note(mc, "Srs   (-2) , Norm = %.14f  E = %.14f", norm_Srs , e_Srs )
    time1 = log.timer('space Srs (-2)', *time1)
    norm_Sij  , e_Sij   = Sij(mc,orbe, dms, eris)
    logger.note(mc, "Sij   (+2) , Norm = %.14f  E = %.14f", norm_Sij , e_Sij )
    time1 = log.timer('space Sij (+2)', *time1)
    norm_Sir  , e_Sir   = Sir(mc,orbe, dms, eris)
    logger.note(mc, "Sir   (0)' , Norm = %.14f  E = %.14f", norm_Sir , e_Sir )
    time1 = log.timer("space Sir (0)'", *time1)

    nevpt_e  = e_Sr + e_Si + e_Sijrs + e_Sijr + e_Srsi + e_Srs + e_Sij + e_Sir
    logger.note(mc, "Nevpt2 Energy = %.15f", nevpt_e)
    log.timer('SC-NEVPT2', *time0)
    return nevpt_e


def _contract4pdm(kern, eri, civec, norb, nelec, link_index=None):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    if link_index is None:
        link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    fdm2 = numpy.empty((norb,norb,norb,norb))
    fdm3 = numpy.empty((norb,norb,norb,norb,norb,norb))
    eri = numpy.ascontiguousarray(eri)

    libmc.NEVPTcontract(ctypes.c_void_p(_ctypes.dlsym(libmc._handle, kern)),
                        fdm2.ctypes.data_as(ctypes.c_void_p),
                        fdm3.ctypes.data_as(ctypes.c_void_p),
                        eri.ctypes.data_as(ctypes.c_void_p),
                        civec.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p))
    for i in range(norb):
        for j in range(i):
            fdm3[j,:,i] = fdm3[i,:,j].transpose(1,0,2,3)
            fdm3[j,i,i,:] += fdm2[j,:]
            fdm3[j,:,i,j] -= fdm2[i,:]
    return fdm3

def _extract_orbs(mc, mo_coeff):
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    mo_vir = mo_coeff[:,nocc:]
    return mo_core, mo_cas, mo_vir

def _norm_to_energy(norm, h, diff):
    idx = abs(norm) > NUMERICAL_ZERO
    ener_t = -(norm[idx] / (diff[idx] + h[idx]/norm[idx])).sum()
    norm_t = norm.sum()
    return norm_t, ener_t

def _ERIS(mc, mo, method='incore'):
    nmo = mo.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nvir = mo.shape[1] - nocc
    mo_cas = mo[:,ncore:nocc]

    if ((method == 'outcore') or
        (mcscf.mc_ao2mo._mem_usage(ncore, ncas, nmo)[0] +
         nmo**4*2/1e6 > mc.max_memory*.9) or
        (mc._scf._eri is None)):
        vhfcore, aapp, appa, apcv, cvcv = \
                trans_e1_outcore(mc, mo, max_memory=mc.max_memory,
                                 verbose=mc.verbose)
    else:
        vhfcore, aapp, appa, apcv, cvcv = \
                trans_e1_incore(mc._scf._eri, mo, mc.ncore, mc.ncas)
    eris = {}
    eris['aapp'] = aapp
    eris['appa'] = appa
    eris['apcv'] = apcv
    eris['cvcv'] = cvcv
    eris['h1eff'] = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore
    return eris

# see mcscf.mc_ao2mo
def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    eri1 = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo), compact=False)

    load_buf = lambda bufid: eri1[bufid*nmo:bufid*nmo+nmo]
    aapp, appa, apcv = _trans_aapp_(mo, ncore, ncas, load_buf)
    vhfcore, cvcv = _trans_cvcv_(mo, ncore, ncas, load_buf)
    return vhfcore, aapp, appa, apcv, cvcv

def trans_e1_outcore(mc, mo, max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=0):
    time0 = (time.clock(), time.time())
    mol = mc.mol
    log = pyscf.lib.logger.Logger(mc.stdout, verbose)
    ncore = mc.ncore
    ncas = mc.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas
    nvir = nmo - nocc

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo), swapfile.name,
                                max_memory=max_memory, ioblk_size=ioblk_size,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(bfn_id):
        if mol.verbose >= pyscf.lib.logger.DEBUG1:
            time1[:] = pyscf.lib.logger.timer(mol, 'between load_buf',
                                              *tuple(time1))
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat[bfn_id*nmo:(bfn_id+1)*nmo]
            col0 = col1
        if mol.verbose >= pyscf.lib.logger.DEBUG1:
            time1[:] = pyscf.lib.logger.timer(mol, 'load_buf', *tuple(time1))
        return buf
    time0 = pyscf.lib.logger.timer(mol, 'halfe1', *time0)
    time1 = [time.clock(), time.time()]
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    aapp, appa, apcv = _trans_aapp_(mo, ncore, ncas, load_buf, ao_loc)
    time0 = pyscf.lib.logger.timer(mol, 'trans_aapp', *time0)
    cvcvfile = tempfile.NamedTemporaryFile()
    with h5py.File(cvcvfile.name) as f5:
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        vhfcore = _trans_cvcv_(mo, ncore, ncas, load_buf, cvcv, ao_loc)[0]
    time0 = pyscf.lib.logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return vhfcore, aapp, appa, apcv, cvcvfile

def _trans_aapp_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    nvir = nmo - nocc
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    klshape = (0, nmo, 0, nmo)

    apcv = numpy.empty((ncas,nmo,ncore,nvir))
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    appa = numpy.empty((ncas,nmo,nmo,ncas))
    ppp = numpy.empty((nmo,nmo,nmo))
    for i in range(ncas):
        buf = _ao2mo.nr_e2_(fload(ncore+i), mo, klshape,
                            aosym='s4', mosym='s2', ao_loc=ao_loc)
        for j in range(nmo):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    ppp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        aapp[i] = ppp[ncore:nocc]
        appa[i] = ppp[:,:,ncore:nocc]
        apcv[i] = ppp[:,:ncore,nocc:]
    return aapp, appa, apcv

def _trans_cvcv_(mo, ncore, ncas, fload, cvcv=None, ao_loc=None):
    nao, nmo = mo.shape
    nocc = ncore + ncas
    nvir = nmo - nocc
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    if cvcv is None:
        cvcv = numpy.zeros((ncore*nvir,ncore*nvir))
    vj = numpy.zeros((nao,nao))
    vk = numpy.zeros((nmo,nao))
    vcv = numpy.empty((nvir,ncore*nvir))
    tmp = numpy.empty((nao,nao))
    for i in range(ncore):
        buf = fload(i)
        funpack(ctypes.c_int(nao), buf[i].ctypes.data_as(ctypes.c_void_p),
                tmp.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        vj += tmp
        for j in range(nmo):
            funpack(ctypes.c_int(nao), buf[j].ctypes.data_as(ctypes.c_void_p),
                    tmp.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
            vk[:,j] += numpy.dot(tmp, mo[:,i])

        klshape = (0, ncore, nocc, nvir)
        _ao2mo.nr_e2_(buf[nocc:nmo], mo, klshape,
                      aosym='s4', mosym='s1', vout=vcv, ao_loc=ao_loc)
        cvcv[i*nvir:(i+1)*nvir] = vcv
    vj = reduce(numpy.dot, (mo.T, vj, mo))
    vk = numpy.dot(mo.T, vk)
    return vj*2-vk, cvcv




if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import mcscf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['O', ( 0., 0.    , 0.    )],
        ['O', ( 0., 0.    , 1.207 )],
    ]
    mol.basis = '6-31g'
    mol.spin = 2
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = mcscf.CASCI(m, 6, 8)
    ci_e = mc.kernel()[0]
    #mc.fcisolver.conv_tol = 1e-14
    mc.verbose = 4
    print(ci_e)
    #dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
    #                                         mc.ci, mc.ci, mc.ncas, mc.nelecas)
    print(sc_nevpt(mc), -0.16978532268234559)


    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 0., 0.    , 0.    )],
        ['H', ( 0., 0.    , 0.8   )],
        ['H', ( 0., 0.    , 2.    )],
        ['H', ( 0., 0.    , 2.8   )],
        ['H', ( 0., 0.    , 4.    )],
        ['H', ( 0., 0.    , 4.8   )],
        ['H', ( 0., 0.    , 6.    )],
        ['H', ( 0., 0.    , 6.8   )],
        ['H', ( 0., 0.    , 8.    )],
        ['H', ( 0., 0.    , 8.8   )],
        ['H', ( 0., 0.    , 10.    )],
        ['H', ( 0., 0.    , 10.8   )],
        ['H', ( 0., 0.    , 12     )],
        ['H', ( 0., 0.    , 12.8   )],
    ]
    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = mcscf.CASCI(m,8,10)
    mc.kernel()
    mc.verbose = 4
    print(sc_nevpt(mc), -0.094164432224406597)
