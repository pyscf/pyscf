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
# Authors: Sheng Guo
#          Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes

import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import fci
from pyscf.mcscf import casci, mc1step, mc_ao2mo
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

libmc = lib.load_library('libmcscf')

NUMERICAL_ZERO = 1e-14
# Ref JCP 117, 9138 (2002); DOI:10.1063/1.1515317

# h1e is the CAS space effective 1e hamiltonian
# h2e is the CAS space 2e integrals in  notation # a' -> p # b' -> q # c' -> r
# d' -> s

def make_a16(h1e, h2e, dms, civec, norb, nelec, link_index=None):
    dm3 = dms['3']
    #dm4 = dms['4']
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
    #dm4 = dms['4']
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


def Sr(mc,ci,dms, eris=None, verbose=None):
    #The subspace S_r^{(-1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    #dm4 = dms['4']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas

    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_cas,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],ncas,ncas,ncas).transpose(0,2,1,3)
        core_dm = numpy.dot(mo_core,mo_core.T) *2
        core_vhf = mc.get_veff(mc.mol,core_dm)
        h1e_v = reduce(numpy.dot, (mo_virt.T, mc.get_hcore()+core_vhf , mo_cas))
        h1e_v -= numpy.einsum('mbbn->mn',h2e_v)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['ppaa'][nocc:,ncore:nocc].transpose(0,2,1,3)
        h1e_v = eris['h1eff'][nocc:,ncore:nocc] - numpy.einsum('mbbn->mn',h2e_v)


    if getattr(mc.fcisolver, 'nevpt_intermediate', None):
        a16 = mc.fcisolver.nevpt_intermediate('A16',ncas,mc.nelecas,ci)
    else:
        a16 = make_a16(h1e,h2e, dms, ci, ncas, mc.nelecas)
    a17 = make_a17(h1e,h2e,dm2,dm3)
    a19 = make_a19(h1e,h2e,dm1,dm2)

    ener = numpy.einsum('ipqr,pqrabc,iabc->i',h2e_v,a16,h2e_v)\
        +  numpy.einsum('ipqr,pqra,ia->i',h2e_v,a17,h1e_v)*2.0\
        +  numpy.einsum('ip,pa,ia->i',h1e_v,a19,h1e_v)

    norm = numpy.einsum('ipqr,rpqbac,iabc->i',h2e_v,dm3,h2e_v)\
        +  numpy.einsum('ipqr,rpqa,ia->i',h2e_v,dm2,h1e_v)*2.0\
        +  numpy.einsum('ip,pa,ia->i',h1e_v,dm1,h1e_v)

    return _norm_to_energy(norm, ener, mc.mo_energy[nocc:])

def Si(mc, ci, dms, eris=None, verbose=None):
    #Subspace S_i^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    #dm4 = dms['4']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas

    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_cas,mo_core,mo_cas,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(ncas,ncore,ncas,ncas).transpose(0,2,1,3)
        core_dm = numpy.dot(mo_core,mo_core.T) *2
        core_vhf = mc.get_veff(mc.mol,core_dm)
        h1e_v = reduce(numpy.dot, (mo_cas.T, mc.get_hcore()+core_vhf , mo_core))
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['ppaa'][ncore:nocc,:ncore].transpose(0,2,1,3)
        h1e_v = eris['h1eff'][ncore:nocc,:ncore]

    if getattr(mc.fcisolver, 'nevpt_intermediate', None):
        #mc.fcisolver.make_a22(ncas, state)
        a22 = mc.fcisolver.nevpt_intermediate('A22',ncas,mc.nelecas,ci)
    else:
        a22 = make_a22(h1e,h2e, dms, ci, ncas, mc.nelecas)
    a23 = make_a23(h1e,h2e,dm1,dm2,dm3)
    a25 = make_a25(h1e,h2e,dm1,dm2)
    delta = numpy.eye(ncas)
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

    return _norm_to_energy(norm, ener, -mc.mo_energy[:ncore])


def Sijrs(mc, eris, verbose=None):
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    ncore = mo_core.shape[1]
    nvirt = mo_virt.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    if eris is None:
        erifile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        feri = ao2mo.outcore.general(mc.mol, (mo_core,mo_virt,mo_core,mo_virt),
                                     erifile.name, verbose=mc.verbose)
    else:
        feri = eris['cvcv']

    eia = mc.mo_energy[:ncore,None] -mc.mo_energy[None,nocc:]
    norm = 0
    e = 0
    with ao2mo.load(feri) as cvcv:
        for i in range(ncore):
            djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
            gi = numpy.asarray(cvcv[i*nvirt:(i+1)*nvirt])
            gi = gi.reshape(nvirt,ncore,nvirt).transpose(1,2,0)
            t2i = (gi.ravel()/djba).reshape(ncore,nvirt,nvirt)
            # 2*ijab-ijba
            theta = gi*2 - gi.transpose(0,2,1)
            norm += numpy.einsum('jab,jab', gi, theta)
            e += numpy.einsum('jab,jab', t2i, theta)
    return norm, e

def Sijr(mc, dms, eris, verbose=None):
    #Subspace S_ijr^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_cas,mo_core],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],ncore,ncas,ncore).transpose(0,2,1,3)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['pacv'][:ncore].transpose(3,1,2,0)
    if 'h1' in dms:
        hdm1 = dms['h1']
    else:
        hdm1 = make_hdm1(dm1)

    a3 = make_a3(h1e,h2e,dm1,dm2,hdm1)
    # We sum norm and h only over i <= j (or j <= i instead).
    # See Eq. (13) and (A2) in https://doi.org/10.1063/1.1515317
    # This implementation is still somewhat wasteful in terms of memory,
    # as we only need about half of norm and h in the end.
    ci_diag = numpy.diag_indices(ncore)
    ci_triu = numpy.triu_indices(ncore)
    norm = 2.0*numpy.einsum('rpji,raji,pa->rji',h2e_v,h2e_v,hdm1)\
         - 1.0*numpy.einsum('rpji,raij,pa->rji',h2e_v,h2e_v,hdm1)
    norm += norm.transpose(0, 2, 1)
    norm[:, ci_diag[0], ci_diag[1]] *= 0.5
    h = 2.0*numpy.einsum('rpji,raji,pa->rji',h2e_v,h2e_v,a3)\
         - 1.0*numpy.einsum('rpji,raij,pa->rji',h2e_v,h2e_v,a3)
    h += h.transpose(0, 2, 1)
    h[:, ci_diag[0], ci_diag[1]] *= 0.5

    diff = mc.mo_energy[nocc:,None,None] - mc.mo_energy[None,:ncore,None] - mc.mo_energy[None,None,:ncore]

    norm_tri = norm[:, ci_triu[0], ci_triu[1]]
    h_tri = h[:, ci_triu[0], ci_triu[1]]
    diff_tri = diff[:, ci_triu[0], ci_triu[1]]
    return _norm_to_energy(norm_tri, h_tri, diff_tri)

def Srsi(mc, dms, eris, verbose=None):
    #Subspace S_ijr^{(1)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    nvirt = mo_virt.shape[1]
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_virt,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],ncore,mo_virt.shape[1],ncas).transpose(0,2,1,3)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['pacv'][nocc:].transpose(3,0,2,1)

    k27 = make_k27(h1e,h2e,dm1,dm2)
    # We sum norm and h only over r <= s.
    # See Eq. (12) and (26) in https://doi.org/10.1063/1.1515317
    # This implementation is still somewhat wasteful in terms of memory,
    # as we only need about half of norm and h in the end.
    vi_diag = numpy.diag_indices(nvirt)
    vi_triu = numpy.triu_indices(nvirt)
    norm = 2.0*numpy.einsum('rsip,rsia,pa->rsi',h2e_v,h2e_v,dm1)\
         - 1.0*numpy.einsum('rsip,sria,pa->rsi',h2e_v,h2e_v,dm1)
    norm += norm.transpose(1, 0, 2)
    norm[vi_diag] *= 0.5
    h = 2.0*numpy.einsum('rsip,rsia,pa->rsi',h2e_v,h2e_v,k27)\
         - 1.0*numpy.einsum('rsip,sria,pa->rsi',h2e_v,h2e_v,k27)
    h += h.transpose(1, 0, 2)
    h[vi_diag] *= 0.5
    diff = mc.mo_energy[nocc:,None,None] + mc.mo_energy[None,nocc:,None] - mc.mo_energy[None,None,:ncore]
    return _norm_to_energy(norm[vi_triu], h[vi_triu], diff[vi_triu])

def Srs(mc, dms, eris=None, verbose=None):
    #Subspace S_rs^{(-2)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    if mo_virt.shape[1] ==0:
        return 0, 0
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_virt,mo_cas],compact=False)
        h2e_v = h2e_v.reshape(mo_virt.shape[1],ncas,mo_virt.shape[1],ncas).transpose(0,2,1,3)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['papa'][nocc:,:,nocc:].transpose(0,2,1,3)

# a7 is very sensitive to the accuracy of HF orbital and CI wfn
    rm2, a7 = make_a7(h1e,h2e,dm1,dm2,dm3)
    norm = 0.5*numpy.einsum('rsqp,rsba,pqba->rs',h2e_v,h2e_v,rm2)
    h = 0.5*numpy.einsum('rsqp,rsba,pqab->rs',h2e_v,h2e_v,a7)
    diff = mc.mo_energy[nocc:,None] + mc.mo_energy[None,nocc:]
    return _norm_to_energy(norm, h, diff)

def Sij(mc, dms, eris, verbose=None):
    #Subspace S_ij^{(-2)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    if mo_core.size ==0 :
        return 0.0, 0
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v = ao2mo.incore.general(mc._scf._eri,[mo_cas,mo_core,mo_cas,mo_core],compact=False)
        h2e_v = h2e_v.reshape(ncas,ncore,ncas,ncore).transpose(0,2,1,3)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v = eris['papa'][:ncore,:,:ncore].transpose(1,3,0,2)

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

# a9 is very sensitive to the accuracy of HF orbital and CI wfn
    a9 = make_a9(h1e,h2e,hdm1,hdm2,hdm3)
    norm = 0.5*numpy.einsum('qpij,baij,pqab->ij',h2e_v,h2e_v,hdm2)
    h = 0.5*numpy.einsum('qpij,baij,pqab->ij',h2e_v,h2e_v,a9)
    diff = mc.mo_energy[:ncore,None] + mc.mo_energy[None,:ncore]
    return _norm_to_energy(norm, h, -diff)


def Sir(mc, dms, eris, verbose=None):
    #Subspace S_il^{(0)}
    mo_core, mo_cas, mo_virt = _extract_orbs(mc, mc.mo_coeff)
    dm1 = dms['1']
    dm2 = dms['2']
    dm3 = dms['3']
    ncore = mo_core.shape[1]
    ncas = mo_cas.shape[1]
    nocc = ncore + ncas
    if eris is None:
        h1e = mc.h1e_for_cas()[0]
        h2e = ao2mo.restore(1, mc.ao2mo(mo_cas), ncas).transpose(0,2,1,3)
        h2e_v1 = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_core,mo_cas,mo_cas],compact=False)
        h2e_v1 = h2e_v1.reshape(mo_virt.shape[1],ncore,ncas,ncas).transpose(0,2,1,3)
        h2e_v2 = ao2mo.incore.general(mc._scf._eri,[mo_virt,mo_cas,mo_cas,mo_core],compact=False)
        h2e_v2 = h2e_v2.reshape(mo_virt.shape[1],ncas,ncas,ncore).transpose(0,2,1,3)
    else:
        h1e = eris['h1eff'][ncore:nocc,ncore:nocc]
        h2e = eris['ppaa'][ncore:nocc,ncore:nocc].transpose(0,2,1,3)
        h2e_v1 = eris['ppaa'][nocc:,:ncore].transpose(0,2,1,3)
        h2e_v2 = eris['papa'][nocc:,:,:ncore].transpose(0,3,1,2)
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
    diff = mc.mo_energy[:ncore,None] - mc.mo_energy[None,nocc:]
    return _norm_to_energy(norm, h, -diff)


class NEVPT(lib.StreamObject):
    '''Strongly contracted NEVPT2

    Attributes:
        root : int
            To control which state to compute if multiple roots or state-average
            wfn were calculated in CASCI/CASSCF
        compressed_mps : bool
            compressed MPS perturber method for DMRG-SC-NEVPT2

    Examples:

    >>> mf = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g').apply(scf.RHF).run()
    >>> mc = mcscf.CASSCF(mf, 4, 4).run()
    >>> NEVPT(mc).kernel()
    -0.14058324991532101
    '''

    _keys = {
        'ncore', 'root', 'compressed_mps', 'e_corr', 'canonicalized', 'onerdm',
    }.union(casci.CASBase._keys, mc1step.CASSCF._keys)

    def __init__(self, mc, root=0):
        self.__dict__.update(mc.__dict__)
        self.ncore = mc.ncore
        self._mc = mc
        self.root = root
        self.compressed_mps = False

##################################################
# don't modify the following attributes, they are not input options
        self.e_corr = None
        self.canonicalized = False
        nao, nmo = mc.mo_coeff.shape
        self.onerdm = numpy.zeros((nao,nao))

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._mc.reset(mol)
        return self

    def get_hcore(self):
        return self._mc.get_hcore()

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=logger.NOTE):
        return self._mc.canonicalize(mo_coeff, ci, eris, sort, cas_natorb, casdm1, verbose)

    def get_veff(self, mol=None, dm=None, hermi=1):
        return self._mc.get_veff(mol, dm, hermi)

    def h1e_for_cas(self, mo_coeff=None, ncas=None, ncore=None):
        return self._mc.h1e_for_cas(mo_coeff, ncas, ncore)

    def load_ci(self, root=None):
        '''Hack me to load CI wfn from disk'''
        if root is None:
            root = self.root
        if self.fcisolver.nroots == 1:
            return self.ci
        else:
            return self.ci[root]

    def for_dmrg(self):
        '''Some preprocess for dmrg-nevpt'''
        if not self._mc.natorb:
            logger.warn(self, '''\
DRMG-MCSCF orbitals are not natural orbitals in active space. It's recommended
to rerun DMRG-CASCI with mc.natorb before calling DMRG-NEVPT2.
See discussions in github issue https://github.com/pyscf/pyscf/issues/698 and
example examples/dmrg/32-dmrg_casscf_nevpt2_for_FeS.py''')
        return self

    def compress_approx(self,maxM=500, nevptsolver=None, tol=1e-7, stored_integral =False):
        '''SC-NEVPT2 with compressed perturber

        Kwargs :
            maxM : int
                DMRG bond dimension

        Examples:

        >>> mf = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g').apply(scf.RHF).run()
        >>> mc = dmrgscf.DMRGSCF(mf, 4, 4).run()
        >>> NEVPT(mc, root=0).compress_approx(maxM=100).kernel()
        -0.14058324991532101

        References:

        J. Chem. Theory Comput. 12, 1583 (2016), doi:10.1021/acs.jctc.5b01225

        J. Chem. Phys. 146, 244102 (2017), doi:10.1063/1.4986975
        '''
        #TODO
        #Some preprocess for compressed perturber
        if getattr(self.fcisolver, 'nevpt_intermediate', None):
            logger.info(self, 'Use compressed mps perturber as an approximation')
        else:
            msg = 'Compressed mps perturber can be only used with DMRG wave function'
            logger.error(self, msg)
            raise RuntimeError(msg)

        self.nevptsolver = nevptsolver
        self.maxM = maxM
        self.tol = tol
        self.stored_integral = stored_integral

        self.canonicalized = True
        self.compressed_mps = True
        self.for_dmrg()
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('NEVPT2 (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info('root = %d', self.root)


    def kernel(self):
        self.dump_flags()
        from pyscf.mcscf.addons import StateAverageFCISolver
        if isinstance(self.fcisolver, StateAverageFCISolver):
            raise RuntimeError('State-average FCI solver object cannot be used '
                               'in NEVPT2 calculation.\nA separated multi-root '
                               'CASCI calculation is required for NEVPT2 method. '
                               'See examples/mrpt/41-for_state_average.py.')

        if getattr(self._mc, 'frozen', None) is not None:
            raise NotImplementedError

        if isinstance(self.verbose, logger.Logger):
            log = self.verbose
        else:
            log = logger.Logger(self.stdout, self.verbose)
        time0 = (logger.process_clock(), logger.perf_counter())
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        #By defaut, _mc is canonicalized for the first root.
        #For SC-NEVPT based on compressed MPS perturber functions, _mc was already canonicalized.
        if (not self.canonicalized):
            # Need to assign roots differently if we have more than one root
            # See issue #1081 (https://github.com/pyscf/pyscf/issues/1081) for more details
            self.mo_coeff, single_ci_vec, self.mo_energy = self.canonicalize(
                self.mo_coeff, ci=self.load_ci(), cas_natorb=True, verbose=self.verbose)
            if self.fcisolver.nroots == 1:
                self.ci = single_ci_vec
            else:
                self.ci[self.root] = single_ci_vec

        if getattr(self.fcisolver, 'nevpt_intermediate', None):
            logger.info(self, 'DMRG-NEVPT')
            dm1, dm2, dm3 = self.fcisolver._make_dm123(self.load_ci(),ncas,self.nelecas,None)
        else:
            dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf',
                                               self.load_ci(), self.load_ci(), ncas, self.nelecas)
        dm4 = None

        dms = {
            '1': dm1, '2': dm2, '3': dm3, '4': dm4,
            # 'h1': hdm1, 'h2': hdm2, 'h3': hdm3
        }
        time1 = log.timer('3pdm, 4pdm', *time0)

        eris = _ERIS(self, self.mo_coeff)
        time1 = log.timer('integral transformation', *time1)

        if not getattr(self.fcisolver, 'nevpt_intermediate', None):  # regular FCI solver
            link_indexa = fci.cistring.gen_linkstr_index(range(ncas), self.nelecas[0])
            link_indexb = fci.cistring.gen_linkstr_index(range(ncas), self.nelecas[1])
            aaaa = eris['ppaa'][ncore:nocc,ncore:nocc].copy()
            f3ca = _contract4pdm('NEVPTkern_cedf_aedf', aaaa, self.load_ci(), ncas,
                                 self.nelecas, (link_indexa,link_indexb))
            f3ac = _contract4pdm('NEVPTkern_aedf_ecdf', aaaa, self.load_ci(), ncas,
                                 self.nelecas, (link_indexa,link_indexb))
            dms['f3ca'] = f3ca
            dms['f3ac'] = f3ac
        time1 = log.timer('eri-4pdm contraction', *time1)

        if self.compressed_mps:
            from pyscf.dmrgscf.nevpt_mpi import DMRG_COMPRESS_NEVPT
            if self.stored_integral: #Stored perturbation integral and read them again. For debugging purpose.
                perturb_file = DMRG_COMPRESS_NEVPT(self, maxM=self.maxM, root=self.root,
                                                   nevptsolver=self.nevptsolver,
                                                   tol=self.tol,
                                                   nevpt_integral='nevpt_perturb_integral')
            else:
                perturb_file = DMRG_COMPRESS_NEVPT(self, maxM=self.maxM, root=self.root,
                                                   nevptsolver=self.nevptsolver,
                                                   tol=self.tol)
            fh5 = h5py.File(perturb_file, 'r')
            e_Si     =   fh5['Vi/energy'][()]
            #The definition of norm changed.
            #However, there is no need to print out it.
            #Only perturbation energy is wanted.
            norm_Si  =   fh5['Vi/norm'][()]
            e_Sr     =   fh5['Vr/energy'][()]
            norm_Sr  =   fh5['Vr/norm'][()]
            fh5.close()
            logger.note(self, "Sr    (-1)',   E = %.14f",  e_Sr  )
            logger.note(self, "Si    (+1)',   E = %.14f",  e_Si  )

        else:
            norm_Sr   , e_Sr    = Sr(self, self.load_ci(), dms, eris)
            logger.note(self, "Sr    (-1)',   E = %.14f",  e_Sr  )
            time1 = log.timer("space Sr (-1)'", *time1)
            norm_Si   , e_Si    = Si(self, self.load_ci(), dms, eris)
            logger.note(self, "Si    (+1)',   E = %.14f",  e_Si  )
            time1 = log.timer("space Si (+1)'", *time1)
        norm_Sijrs, e_Sijrs = Sijrs(self, eris)
        logger.note(self, "Sijrs (0)  ,   E = %.14f", e_Sijrs)
        time1 = log.timer('space Sijrs (0)', *time1)
        norm_Sijr , e_Sijr  = Sijr(self, dms, eris)
        logger.note(self, "Sijr  (+1) ,   E = %.14f",  e_Sijr)
        time1 = log.timer('space Sijr (+1)', *time1)
        norm_Srsi , e_Srsi  = Srsi(self, dms, eris)
        logger.note(self, "Srsi  (-1) ,   E = %.14f",  e_Srsi)
        time1 = log.timer('space Srsi (-1)', *time1)
        norm_Srs  , e_Srs   = Srs(self, dms, eris)
        logger.note(self, "Srs   (-2) ,   E = %.14f",  e_Srs )
        time1 = log.timer('space Srs (-2)', *time1)
        norm_Sij  , e_Sij   = Sij(self, dms, eris)
        logger.note(self, "Sij   (+2) ,   E = %.14f",  e_Sij )
        time1 = log.timer('space Sij (+2)', *time1)
        norm_Sir  , e_Sir   = Sir(self, dms, eris)
        logger.note(self, "Sir   (0)' ,   E = %.14f",  e_Sir )
        time1 = log.timer("space Sir (0)'", *time1)

        nevpt_e  = e_Sr + e_Si + e_Sijrs + e_Sijr + e_Srsi + e_Srs + e_Sij + e_Sir
        logger.note(self, "Nevpt2 Energy = %.15f", nevpt_e)
        log.timer('SC-NEVPT2', *time0)

        self.e_corr = nevpt_e
        return nevpt_e


def kernel(mc, *args, **kwargs):
    return sc_nevpt(mc, *args, **kwargs)

def sc_nevpt(mc, ci=None, verbose=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("once")
        warnings.warn('API updates: function sc_nevpt is deprecated feature. '
                      'It will be removed in future release.\n'
                      'It is recommended to run NEVPT2 with new function '
                      'mrpt.NEVPT(mc).kernel()')
        if ci is not None:
            warnings.warn('API updates: The kwarg "ci" has no effects. '
                          'Use mrpt.NEVPT(mc,root=?) for excited state.')
    return NEVPT(mc).kernel()


# register NEVPT2 in MCSCF
casci.CASBase.NEVPT2 = NEVPT







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

    libmc.NEVPTcontract(getattr(libmc, kern),
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

    mem_incore, mem_outcore, mem_basic = mc_ao2mo._mem_usage(ncore, ncas, nmo)
    mem_now = lib.current_memory()[0]
    if (method == 'incore' and mc._scf._eri is not None and
        (mem_incore+mem_now < mc.max_memory*.9) or
        mc.mol.incore_anyway):
        ppaa, papa, pacv, cvcv = trans_e1_incore(mc, mo)
    else:
        max_memory = max(2000, mc.max_memory-mem_now)
        ppaa, papa, pacv, cvcv = \
                trans_e1_outcore(mc, mo, max_memory=max_memory,
                                 verbose=mc.verbose)

    dmcore = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj*2-vk, mo))

    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore
    return eris

# see mcscf.mc_ao2mo
def trans_e1_incore(mc, mo):
    eri_ao = mc._scf._eri
    ncore = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nocc = ncore + ncas
    nav = nmo - ncore
    eri1 = ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo[:,ncore:]),
                                compact=False)
    load_buf = lambda r0,r1: eri1[r0*nav:r1*nav]
    ppaa, papa, pacv, cvcv = _trans(mo, ncore, ncas, load_buf)
    return ppaa, papa, pacv, cvcv

def trans_e1_outcore(mc, mo, max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=0):
    time0 = (logger.process_clock(), logger.perf_counter())
    mol = mc.mol
    log = logger.Logger(mc.stdout, verbose)
    ncore = mc.ncore
    ncas = mc.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas
    nvir = nmo - nocc
    nav = nmo - ncore

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo[:,ncore:]), swapfile.name,
                          max_memory=max_memory, ioblk_size=ioblk_size,
                          verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(r0,r1):
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'between load_buf',
                                              *tuple(time1))
        buf = numpy.empty(((r1-r0)*nav,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:,col0:col1] = dat[r0*nav:r1*nav]
            col0 = col1
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'load_buf', *tuple(time1))
        return buf
    time0 = logger.timer(mol, 'halfe1', *time0)
    time1 = [logger.process_clock(), logger.perf_counter()]
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    cvcvfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    with lib.H5TmpFile(cvcvfile.name, 'w') as f5:
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        ppaa, papa, pacv = _trans(mo, ncore, ncas, load_buf, cvcv, ao_loc)[:3]
    time0 = logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return ppaa, papa, pacv, cvcvfile

def _trans(mo, ncore, ncas, fload, cvcv=None, ao_loc=None):
    nao, nmo = mo.shape
    nocc = ncore + ncas
    nvir = nmo - nocc
    nav = nmo - ncore

    if cvcv is None:
        cvcv = numpy.zeros((ncore*nvir,ncore*nvir))
    pacv = numpy.empty((nmo,ncas,ncore*nvir))
    aapp = numpy.empty((ncas,ncas,nmo*nmo))
    papa = numpy.empty((nmo,ncas,nmo*ncas))
    vcv = numpy.empty((nav,ncore*nvir))
    apa = numpy.empty((ncas,nmo*ncas))
    vpa = numpy.empty((nav,nmo*ncas))
    app = numpy.empty((ncas,nmo*nmo))
    for i in range(ncore):
        buf = fload(i, i+1)
        klshape = (0, ncore, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        cvcv[i*nvir:(i+1)*nvir] = vcv[ncas:]
        pacv[i] = vcv[:ncas]

        klshape = (0, nmo, ncore, nocc)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=apa, ao_loc=ao_loc)
        papa[i] = apa
    for i in range(ncas):
        buf = fload(ncore+i, ncore+i+1)
        klshape = (0, ncore, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        pacv[ncore:,i] = vcv

        klshape = (0, nmo, ncore, nocc)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vpa, ao_loc=ao_loc)
        papa[ncore:,i] = vpa

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=app, ao_loc=ao_loc)
        aapp[i] = app
    ppaa = lib.transpose(aapp.reshape(ncas**2,-1))
    return (ppaa.reshape(nmo,nmo,ncas,ncas), papa.reshape(nmo,ncas,nmo,ncas),
            pacv.reshape(nmo,ncas,ncore,nvir), cvcv)




if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
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
    m.conv_tol = 1e-20
    ehf = m.scf()
    mc = mcscf.CASCI(m, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    ci_e = mc.kernel()[0]
    mc.verbose = 4
    print(ci_e)
    #dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
    #                                         mc.ci, mc.ci, mc.ncas, mc.nelecas)
    print(sc_nevpt(mc), -0.169785157128082)


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
    m.conv_tol = 1e-20
    ehf = m.scf()
    mc = mcscf.CASCI(m,8,10)
    mc.fcisolver.conv_tol = 1e-14
    mc.kernel()
    mc.verbose = 4
    print(sc_nevpt(mc), -0.094164359938171)
