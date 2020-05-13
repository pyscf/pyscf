# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import time
import pyscf.pbc.tools as tools
from mpi4py import MPI
from pyscf.lib import logger
from pyscf.pbc.mpitools import mpi_load_balancer
from pyscf import lib
from pyscf.pbc import lib as pbclib
from pyscf.pbc.tools.tril import tril_index, unpack_tril

comm = MPI.COMM_WORLD

#einsum = np.einsum
einsum = lib.einsum
dot = np.dot

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_tau1(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    tmp_oovv_shape = BLKSIZE + (nocc,nocc,nvir,nvir)
    tmp_oovv  = np.empty(tmp_oovv_shape,dtype=t1.dtype)
    tmp2_oovv = np.zeros_like(tmp_oovv)

    tau1_ooVv = feri2['tau1_ooVv']
    tau1_oOvv = feri2['tau1_oOvv']
    tau1_oovv_rev = feri2['tau1_oovv_rev']
    tau2_Oovv = feri2['tau2_Oovv']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterki, ki in enumerate(ranges0):
            for iterkj, kj in enumerate(ranges1):
                for iterka, ka in enumerate(ranges2):
                    kb = kconserv[ki,ka,kj]
                    tmp_oovv[iterki,iterkj,iterka] = t2[ki,kj,ka].copy()
                    tmp2_oovv[iterki,iterkj,iterka] *= 0.0
                    if ki == ka and kj == kb:
                        tmp2_oovv[iterki,iterkj,iterka] = einsum('ia,jb->ijab',t1[ki],t1[kj])

                    tau1_oovv_rev[kj,ka,kb] = (tmp_oovv[iterki,iterkj,iterka] + tmp2_oovv[iterki,iterkj,iterka])

        tau1_ooVv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,nvir*min(ranges2):nvir*(max(ranges2)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(0,1,2,5,3,4,6).reshape(len(ranges0),len(ranges1),len(ranges2)*nvir,nocc,nocc,nvir)
        tau1_oOvv[min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges1):nocc*(max(ranges1)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(0,2,1,4,3,5,6).reshape(len(ranges0),len(ranges2),len(ranges1)*nocc,nocc,nvir,nvir)
        tau2_Oovv[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges0):nocc*(max(ranges0)+1)] = \
                ( tmp_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] +
                        2*tmp2_oovv[:len(ranges0),:len(ranges1),:len(ranges2)] ).transpose(1,2,0,3,4,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nocc,nvir,nvir)
        loader.slave_finished()
    comm.Barrier()
    return

def cc_Foo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    Fki = np.empty((nkpts,nocc,nocc),dtype=t2.dtype)

    for ki in range(nkpts):
        kk = ki
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()
        for kl in range(nkpts):
            for kc in range(nkpts):
            #Fki[ki] += einsum('lkcd,licd->ki',eris.SoOvv[kk,kc],tau1_oOvv[ki,kc])
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                #Fki[ki] += einsum('klcd,ilcd->ki',Soovv,t2[ki,kl,kc])
                Fki[ki] += einsum('klcd,ilcd->ki',Soovv,unpack_tril(t2,nkpts,ki,kl,kc,kconserv[ki,kc,kl]))
            #if ki == kc:
            kd = kconserv[kk,ki,kl]
            Soovv = 2*eris.oovv[kk,kl,ki] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
            Fki[ki] += einsum('klcd,ic,ld->ki',Soovv,t1[ki],t1[kl])
    return Fki

def cc_Fvv(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    Fac = np.empty((nkpts,nvir,nvir),dtype=t2.dtype)

    for ka in range(nkpts):
        kc = ka
        Fac[ka] = eris.fock[ka,nocc:,nocc:].copy()
        #for kk in range(nkpts):
        #    Fac[ka] += -einsum('lkcd,lkad->ac',eris.SoOvv[kk,kc],tau1_oOvv[kk,ka])
        for kl in range(nkpts):
            for kk in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                #Fac[ka] += -einsum('klcd,klad->ac',Soovv,t2[kk,kl,ka])
                Fac[ka] += -einsum('klcd,klad->ac',Soovv,unpack_tril(t2,nkpts,kk,kl,ka,kconserv[kk,ka,kl]))
            #if kk == ka
            kd = kconserv[ka,kc,kl]
            Soovv = 2*eris.oovv[ka,kl,kc] - eris.oovv[ka,kl,kd].transpose(0,1,3,2)
            Fac[ka] += -einsum('klcd,ka,ld->ac',Soovv,t1[ka],t1[kl])
    return Fac

def cc_Fov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    Fkc = np.empty((nkpts,nocc,nvir),dtype=t2.dtype)
    Fkc[:] = eris.fock[:,:nocc,nocc:].copy()
    for kk in range(nkpts):
        for kl in range(nkpts):
            Soovv = 2.*eris.oovv[kk,kl,kk] - eris.oovv[kk,kl,kl].transpose(0,1,3,2)
            Fkc[kk] += einsum('klcd,ld->kc',Soovv,t1[kl])
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(cc,t1,t2,eris,feri2)
    for ki in range(nkpts):
        Lki[ki] += einsum('kc,ic->ki',fov[ki],t1[ki])
        SoOov = (2*eris.ooov[ki,:,ki] - eris.ooov[:,ki,ki].transpose(0,2,1,3,4)).transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nocc,nvir)
        Lki[ki] += einsum('lkic,lc->ki',SoOov,t1.reshape(nkpts*nocc,nvir))
    return Lki

def Lvv(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(cc,t1,t2,eris,feri2)
    for ka in range(nkpts):
        Lac[ka] += -einsum('kc,ka->ac',fov[ka],t1[ka])
        for kk in range(nkpts):
            Svovv = 2*eris.ovvv[kk,ka,kk].transpose(1,0,3,2) - eris.ovvv[kk,ka,ka].transpose(1,0,2,3)
            Lac[ka] += einsum('akcd,kd->ac',Svovv,t1[kk])
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    khelper = cc.khelper

    #Wklij = np.array(eris.oooo, copy=True)
    #for pqr in range(nUnique_klist):
    #    kk, kl, ki = unique_klist[pqr]
    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    oooo_tmp_shape = BLKSIZE + (nocc,nocc,nocc,nocc)
    oooo_tmp = np.empty(shape=oooo_tmp_shape,dtype=t1.dtype)

    tau1_ooVv = feri2['tau1_ooVv']
    #Woooo     = feri2['Woooo']
    Woooo_rev = feri2['Woooo_rev']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterkk, kk in enumerate(ranges0):
            for iterkl, kl in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kj = kconserv[kk,ki,kl]
                    oooo_tmp[iterkk,iterkl,iterki] = np.array(eris.oooo[kk,kl,ki],copy=True)
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klcj,ic->klij',eris.ooov[kl,kk,kj].transpose(1,0,3,2),t1[ki])

                    # Note the indices and way the tau1 is stored : instead of a loop over kpt='kc' and
                    # loop over mo='c', the (kc,k,l,c,d) index is changed instead to (nkpts*nvir,k,l,d)
                    # so that we only have to loop over the first index, saving read operations.
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('ckld,cijd->klij',eris.ooVv[kk,kl],tau1_ooVv[ki,kj])

                    #for kc in range(nkpts):
                    #    oooo_tmp[iterkk,iterkl,iterki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                    #oooo_tmp[iterkk,iterkl,iterki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                    #Woooo[kk,kl,ki] = oooo_tmp[iterkk,iterkl,iterki]
                    #Woooo[kl,kk,kj] = oooo_tmp[iterkk,iterkl,iterki].transpose(1,0,3,2)
                    Woooo_rev[kl,ki,kj] = oooo_tmp[iterkk,iterkl,iterki]

        #Woooo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #                oooo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]

        # for if you want to take into account symmetry of Woooo integral
        #feri2.Woooo[min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1] = \
        #                oooo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,1,2,4,3,6,5)
        loader.slave_finished()
    comm.Barrier()
    return

def cc_Wvvvv(cc,t1,t2,eris,feri2=None):
    ## Slow:
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    vvvv_tmp = np.empty((nvir,nvir,nvir,nvir),dtype=t1.dtype)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,))
    loader.set_ranges((range(nkpts),range(nkpts),))

    Wvvvv = feri2['Wvvvv']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1 = loader.get_blocks_from_data(data)
        for ka in ranges0:
            for kc in ranges1:
                for kb in range(ka+1):
                    kd = kconserv[ka,kc,kb]
                    vvvv_tmp = np.array(eris.vvvv[ka,kb,kc],copy=True)
                    vvvv_tmp += einsum('akcd,kb->abcd',eris.ovvv[kb,ka,kd].transpose(1,0,3,2),-t1[kb])
                    vvvv_tmp += einsum('kbcd,ka->abcd',eris.ovvv[ka,kb,kc],-t1[ka])
                    Wvvvv[ka,kb,kc] = vvvv_tmp
                    Wvvvv[kb,ka,kd] = vvvv_tmp.transpose(1,0,3,2)
        loader.slave_finished()

    ## Fast
    #nocc,nvir = t1.shape
    #Wabcd = np.empty((nvir,)*4)
    #for a in range(nvir):
    #    Wabcd[a,:] = einsum('kcd,kb->bcd',eris.vovv[a],-t1)
    ##Wabcd += einsum('kbcd,ka->abcd',eris.ovvv,-t1)
    #Wabcd += lib.dot(-t1.T,eris.ovvv.reshape(nocc,-1)).reshape((nvir,)*4)
    #Wabcd += np.asarray(eris.vvvv)

    comm.Barrier()
    return

def cc_Wvoov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    #Wakic = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    voov_tmp_shape = BLKSIZE + (nvir,nocc,nocc,nvir)
    voov_tmp = np.empty(voov_tmp_shape,dtype=t1.dtype)

    tau2_Oovv = feri2['tau2_Oovv']
    #Wvoov     = feri2['Wvoov']
    WvOov     = feri2['WvOov']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        ix = sum([[min(x),max(x)+1] for x in (ranges0,ranges1,ranges2)], [])

        #eris_ooov = eris.ooov[ix[0]:ix[1], ix[2]:ix[3], ix[4]:ix[5]]
        for iterka, ka in enumerate(ranges0):
            for iterkk, kk in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kc = kconserv[ka,ki,kk]
                    voov_tmp[iterka,iterkk,iterki] = np.array(eris.ovvo[kk,ka,kc]).transpose(1,0,3,2)
                    voov_tmp[iterka,iterkk,iterki] -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
                    voov_tmp[iterka,iterkk,iterki] += einsum('akdc,id->akic',eris.ovvv[kk,ka,kc].transpose(1,0,3,2),t1[ki])
                    # Beginning of change
                    #for kl in range(nkpts):
                    #    # kl - kd + kk = kc
                    #    # => kd = kl - kc + kk
                    #    kd = kconserv[kl,kc,kk]
                    #    Soovv = 2*np.array(eris.oovv[kl,kk,kd]) - np.array(eris.oovv[kl,kk,kc]).transpose(0,1,3,2)
                    #    voov_tmp[iterka,iterkk,iterki] += 0.5*einsum('lkdc,ilad->akic',Soovv,t2[ki,kl,ka])
                    #    voov_tmp[iterka,iterkk,iterki] -= 0.5*einsum('lkdc,ilda->akic',eris.oovv[kl,kk,kd],t2[ki,kl,kd])
                    #voov_tmp[iterka,iterkk,iterki] -= einsum('lkdc,id,la->akic',eris.oovv[ka,kk,ki],t1[ki],t1[ka])
                    #Wvoov[ka,kk,ki] = voov_tmp[iterka,iterkk,iterki]

                    # Making various intermediates...
                    #t2_oOvv = t2[ki,:,ka].transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)
                    t2_oOvv = unpack_tril(t2,nkpts,ki,range(nkpts),ka,kconserv[ki,ka,range(nkpts)]).transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)
                    #eris_oOvv = eris.oovv[kk,:,kc].transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)

                    voov_tmp[iterka,iterkk,iterki] += 0.5*einsum('lkcd,liad->akic',eris.SoOvv[kk,kc],t2_oOvv)
                    voov_tmp[iterka,iterkk,iterki] -= 0.5*einsum('lkcd,liad->akic',eris.oOvv[kk,kc],tau2_Oovv[ki,ka])

                    # End of change
        #Wvoov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #        voov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WvOov[min(ranges0):max(ranges0)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges1):nocc*(max(ranges1)+1)] = \
                voov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,2,1,4,3,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nvir,nocc,nvir)
        loader.slave_finished()
    comm.Barrier()
    return

def cc_Wvovo(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    vovo_tmp_shape = BLKSIZE + (nvir,nocc,nvir,nocc)
    vovo_tmp = np.empty(shape=vovo_tmp_shape,dtype=t1.dtype)

    Wvovo = feri2['Wvovo']
    WvOVo = feri2['WvOVo']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterka, ka in enumerate(ranges0):
            for iterkk, kk in enumerate(ranges1):
                for iterkc, kc in enumerate(ranges2):
                    ki = kconserv[ka,kc,kk]
                    vovo_tmp[iterka,iterkk,iterkc] = np.array(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)
                    vovo_tmp[iterka,iterkk,iterkc] -= einsum('lkci,la->akci',eris.ooov[kk,ka,ki].transpose(1,0,3,2),t1[ka])
                    vovo_tmp[iterka,iterkk,iterkc] += einsum('akcd,id->akci',eris.ovvv[kk,ka,ki].transpose(1,0,3,2),t1[ki])
                    # Beginning of change
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    vovo_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                    #vovo_tmp[iterka,iterkk,iterkc] -= einsum('lkcd,id,la->akci',eris.oovv[ka,kk,kc],t1[ki],t1[ka])
                    #Wvovo[ka,kk,kc] = vovo_tmp[iterka,iterkk,iterkc]

                    oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
                    #t2f   = t2[:,ki,ka].copy() #This is a tau like term
                    t2f   = unpack_tril(t2,nkpts,range(nkpts),ki,ka,kconserv[range(nkpts),ka,ki]).copy() #This is a tau like term
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    if ki == kd and kl == ka:
                    #        t2f[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    kd = kconserv[ka,kc,kk]
                    t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                    t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
                    vovo_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)

        Wvovo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                vovo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WvOVo[min(ranges0):max(ranges0)+1,nocc*min(ranges1):nocc*(max(ranges1)+1),nvir*min(ranges2):nvir*(max(ranges2)+1)] = \
                vovo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(0,1,4,2,5,3,6).reshape(len(ranges0),len(ranges1)*nocc,len(ranges2)*nvir,nvir,nocc)
                    # End of change
        loader.slave_finished()
    comm.Barrier()
    return

def cc_Wovov(cc,t1,t2,eris,feri2=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    BLKSIZE = (1,1,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    ovov_tmp_shape = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(shape=ovov_tmp_shape,dtype=t1.dtype)

    #Wovov = feri2['Wovov']
    WOvov = feri2['WOvov']

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        for iterkk, kk in enumerate(ranges0):
            for iterka, ka in enumerate(ranges1):
                for iterki, ki in enumerate(ranges2):
                    kc = kconserv[kk,ki,ka]
                    ovov_tmp[iterkk,iterka,iterki] = np.array(eris.ovov[kk,ka,ki],copy=True)
                    ovov_tmp[iterkk,iterka,iterki] -= einsum('lkci,la->kaic',eris.ooov[kk,ka,ki].transpose(1,0,3,2),t1[ka])
                    ovov_tmp[iterkk,iterka,iterki] += einsum('akcd,id->kaic',eris.ovvv[kk,ka,ki].transpose(1,0,3,2),t1[ki])
                    # Beginning of change
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    ovov_tmp[iterka,iterkk,iterkc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                    #ovov_tmp[iterka,iterkk,iterkc] -= einsum('lkcd,id,la->akci',eris.oovv[ka,kk,kc],t1[ki],t1[ka])
                    #Wvovo[ka,kk,kc] = ovov_tmp[iterka,iterkk,iterkc]

                    oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
                    #t2f   = t2[:,ki,ka].copy() #This is a tau like term
                    t2f   = unpack_tril(t2,nkpts,range(nkpts),ki,ka,kconserv[range(nkpts),ka,ki]).copy() #This is a tau like term
                    #for kl in range(nkpts):
                    #    kd = kconserv[kl,kc,kk]
                    #    if ki == kd and kl == ka:
                    #        t2f[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    kd = kconserv[ka,kc,kk]
                    t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                    t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
                    ovov_tmp[iterkk,iterka,iterki] -= 0.5*einsum('lkcd,liad->kaic',oovvf,t2f)

        #Wovov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
        #        ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        WOvov[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,nocc*min(ranges0):nocc*(max(ranges0)+1)] = \
                ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(1,2,0,3,4,5,6).reshape(len(ranges1),len(ranges2),len(ranges0)*nocc,nvir,nocc,nvir)
                    # End of change
        loader.slave_finished()
    comm.Barrier()
    return

# EOM Intermediates w/ k-points

def Wooov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wklid = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir),dtype=t2.dtype)
    else:
        Wklid = fint['Wooov']

    # TODO can do much better than this... call recursive function
    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ooov_tmp_size = BLKSIZE + (nocc,nocc,nocc,nvir)
    ooov_tmp = np.empty(ooov_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_ooov_kli = _cp(eris.ooov[s0,s1,s2])
        eris_oovv_kli = _cp(eris.oovv[s0,s1,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkl,kl in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kd = kconserv[kk,ki,kl]
                    ooov_tmp[iterkk,iterkl,iterki] = eris_ooov_kli[iterkk,iterkl,iterki].copy()
                    ooov_tmp[iterkk,iterkl,iterki] += einsum('ic,klcd->klid',t1[ki],eris_oovv_kli[iterkk,iterkl,iterki])
        Wklid[s0,s1,s2] = ooov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wklid, op=MPI.SUM)

    return Wklid


def Wvovv(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Walcd = np.zeros((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir),dtype=t2.dtype)
    else:
        Walcd = fint['Wvovv']

    # TODO can do much better than this... call recursive function
    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nvir*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    vovv_tmp_size = BLKSIZE + (nvir,nocc,nvir,nvir)
    vovv_tmp = np.empty(vovv_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_vovv_alc = _cp(eris.vovv[s0,s1,s2])
        eris_oovv_alc = _cp(eris.oovv[s0,s1,s2])

        for iterka,ka in enumerate(ranges0):
            for iterkl,kl in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    kd = kconserv[ka,kc,kl]
                    # vovv[ka,kl,kc,kd] <= ovvv[kl,ka,kd,kc].transpose(1,0,3,2)
                    vovv_tmp[iterka,iterkl,iterkc] = eris_vovv_alc[iterka,iterkl,iterkc] #np.array(eris.ovvv[kl,ka,kd]).transpose(1,0,3,2)
                    vovv_tmp[iterka,iterkl,iterkc] += -einsum('ka,klcd->alcd',t1[ka],eris_oovv_alc[iterka,iterkl,iterkc])
        Walcd[s0,s1,s2] = vovv_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Walcd, op=MPI.SUM)

    return Walcd

def W1ovvo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    else:
        Wkaci  = fint['W1ovvo']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovvo_tmp_size = BLKSIZE + (nocc,nvir,nvir,nocc)
    ovvo_tmp = np.empty(ovvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        eris_ovvo_kac = _cp(eris.ovvo[s0,s1,s2])
        eris_oovv_kXc = _cp(eris.oovv[s0,:,s2])
        eris_oovv_Xkc = _cp(eris.oovv[:,s0,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterka,ka in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    ki = kconserv[kk,kc,ka]
                    ovvo_tmp[iterkk,iterka,iterkc] = _cp(eris_ovvo_kac[iterkk,iterka,iterkc])
                    #St2 = 2.*t2[ki,:,ka]
                    St2 = 2.*unpack_tril(t2,nkpts,ki,range(nkpts),ka,kconserv[ki,ka,range(nkpts)])
                    #St2 -= t2[:,ki,ka].transpose(0,2,1,3,4)
                    St2 -= unpack_tril(t2,nkpts,range(nkpts),ki,ka,kconserv[range(nkpts),ka,ki]).transpose(0,2,1,3,4)
                    ovvo_tmp[iterkk,iterka,iterkc] += einsum('klcd,ilad->kaci',eris_oovv_kXc[iterkk,:,iterkc].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir),
                                                                St2.transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    ovvo_tmp[iterkk,iterka,iterkc] += -einsum('lkcd,ilad->kaci',eris_oovv_Xkc[:,iterkk,iterkc].reshape(nocc*nkpts,nocc,nvir,nvir),
                                               unpack_tril(t2,nkpts,ki,range(nkpts),ka,kconserv[ki,ka,range(nkpts)]).transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
#                                                                t2[ki,:,ka].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
        Wkaci[s0,s1,s2] = ovvo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci

def W2ovvo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
        WWooov = Wooov(cc,t1,t2,eris)
    else:
        Wkaci  = fint['W2ovvo']
        WWooov = fint['Wooov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovvo_tmp_size = BLKSIZE + (nocc,nvir,nvir,nocc)
    ovvo_tmp = np.empty(ovvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        Wooov_akX     = _cp(WWooov[s1,s0])
        eris_ovvv_kac = _cp(eris.ovvv[s0,s1,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterka,ka in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    ki = kconserv[kk,kc,ka]
                    ovvo_tmp[iterkk,iterka,iterkc] = einsum('la,lkic->kaci',-t1[ka],Wooov_akX[iterka,iterkk,ki])
                    ovvo_tmp[iterkk,iterka,iterkc] += einsum('akdc,id->kaci',eris_ovvv_kac[iterkk,iterka,iterkc].transpose(1,0,3,2),t1[ki])

        Wkaci[s0,s1,s2] = ovvo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci


def Wovvo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t2.dtype)
        W1kaci = W1ovvo(cc,t1,t2,eris,fint)
        W2kaci = W2ovvo(cc,t1,t2,eris,fint)
    else:
        Wkaci = fint['Wovvo']
        W1kaci = fint['W1ovvo']
        W2kaci = fint['W2ovvo']

    # TODO can do much better than this... call recursive function
    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        Wkaci[s0,s1,s2] = _cp(W1kaci[s0,s1,s2]) + _cp(W2kaci[s0,s1,s2])

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci

def W1ovov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkbid = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t2.dtype)
    else:
        Wkbid = fint['W1ovov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovov_tmp_size = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(ovov_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_ovov = _cp(eris.ovov[s0,s1,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kd = kconserv[kk,ki,kb]
                    ovov_tmp[iterkk,iterkb,iterki] = eris_ovov[iterkk,iterkb,iterki].copy()
                    ovov_tmp[iterkk,iterkb,iterki] += -einsum('lkdc,libc->kbid',eris.oovv[:,kk,kd].reshape(nkpts*nocc,nocc,nvir,nvir),
#                                                              t2[:,ki,kb].reshape(nkpts*nocc,nocc,nvir,nvir))
                                                              unpack_tril(t2,nkpts,range(nkpts),ki,kb,kconserv[range(nkpts),kb,ki]).reshape(nkpts*nocc,nocc,nvir,nvir))
        Wkbid[s0,s1,s2] = ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkbid, op=MPI.SUM)

    return Wkbid

def W2ovov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkbid = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t2.dtype)
        WWooov = Wooov(cc,t1,t2,eris)
    else:
        Wkbid = fint['W2ovov']
        WWooov = fint['Wooov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovov_tmp_size = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(ovov_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_ovvv  = _cp(eris.ovvv[s0,s1,s2])
        WWooov_kbi = _cp(WWooov[s0,s1,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kd = kconserv[kk,ki,kb]
                    ovov_tmp[iterkk,iterkb,iterki] = einsum('klid,lb->kbid',WWooov_kbi[iterkk,iterkb,iterki],-t1[kb])
                    ovov_tmp[iterkk,iterkb,iterki] += einsum('kbcd,ic->kbid',eris_ovvv[iterkk,iterkb,iterki],t1[ki])
        Wkbid[s0,s1,s2] = ovov_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkbid, op=MPI.SUM)

    return Wkbid

def Wovov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkbid = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t2.dtype)
        WW1ovov = W1ovov(cc,t1,t2,eris)
        WW2ovov = W2ovov(cc,t1,t2,eris)
    else:
        Wkbid = fint['Wovov']
        WW1ovov = fint['W1ovov']
        WW2ovov = fint['W2ovov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovov_tmp_size = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(ovov_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        Wkbid[s0,s1,s2] = _cp(WW1ovov[s0,s1,s2]) + _cp(WW2ovov[s0,s1,s2])

        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkbid, op=MPI.SUM)

    return Wkbid


def WovovRev(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkbid = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t2.dtype)
        WW1ovov = W1ovov(cc,t1,t2,eris)
        WW2ovov = W2ovov(cc,t1,t2,eris)
    else:
        Wkbid = fint['WovovRev']
        WW1ovov = fint['W1ovov']
        WW2ovov = fint['W2ovov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovov_tmp_size = BLKSIZE + (nocc,nvir,nocc,nvir)
    ovov_tmp = np.empty(ovov_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        Wkbid[s2,s1,s0] = (_cp(WW1ovov[s0,s1,s2]) + _cp(WW2ovov[s0,s1,s2])).transpose(2,1,0,3,4,5,6)

        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkbid, op=MPI.SUM)

    return Wkbid



# This is the same Woooo intermediate used in cc

def Woooo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wklij = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc),dtype=t2.dtype)
    else:
        Wklij = fint['Woooo']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    nkpts_blksize2 = 1
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    oooo_tmp_size = BLKSIZE + (nocc,nocc,nocc,nocc)
    oooo_tmp = np.empty(oooo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_oovv_klX = _cp(eris.oovv[s0,s1,s2])
        eris_oooo_kli = _cp(eris.oooo[s0,s1,s2])
        eris_ooov_klX = _cp(eris.ooov[s0,s1,s2])
        eris_ooov_lkX = _cp(eris.ooov[s1,s0,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkl,kl in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kj = kconserv[kk,ki,kl]
                    #tau1 = t2[ki,kj,:].copy()
                    tau1 = unpack_tril(t2,nkpts,ki,kj,range(nkpts),kconserv[ki,range(nkpts),kj]).copy()
                    tau1[ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] = eris_oooo_kli[iterkk,iterkl,iterki].copy()
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('kld,ijd->klij',eris_oovv_klX[iterkk,iterkl,:].transpose(1,2,0,3,4).reshape(nocc,nocc,-1),
                                                                 tau1.transpose(1,2,0,3,4).reshape(nocc,nocc,-1))
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klid,jd->klij',eris_ooov_klX[iterkk,iterkl,ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('lkjc,ic->klij',eris_ooov_lkX[iterkl,iterkk,kj],t1[ki])
        Wklij[s0,s1,s2] = oooo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wklij, op=MPI.SUM)

    return Wklij

# This has different storage compared to Woooo, more amenable to I/O
# Instead of calling Woooo[kk,kl,ki] to get Woooo[kk,kl,ki,kj] you call
# WooooS[kl,ki,kj]
def WooooS(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wklij = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc),dtype=t2.dtype)
    else:
        Wklij = fint['WooooS']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    nkpts_blksize2 = 1
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    oooo_tmp_size = BLKSIZE + (nocc,nocc,nocc,nocc)
    oooo_tmp = np.empty(oooo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_oovv_klX = _cp(eris.oovv[s0,s1,s2])
        eris_oooo_kli = _cp(eris.oooo[s0,s1,s2])
        eris_ooov_klX = _cp(eris.ooov[s0,s1,s2])
        eris_ooov_lkX = _cp(eris.ooov[s1,s0,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkl,kl in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kj = kconserv[kk,ki,kl]
                    #tau1 = t2[ki,kj,:].copy()
                    tau1 = unpack_tril(t2,nkpts,ki,kj,range(nkpts),kconserv[ki,range(nkpts),kj]).copy()
                    tau1[ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] = eris_oooo_kli[iterkk,iterkl,iterki].copy()
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('kld,ijd->klij',eris_oovv_klX[iterkk,iterkl,:].transpose(1,2,0,3,4).reshape(nocc,nocc,-1),
                                                                 tau1.transpose(1,2,0,3,4).reshape(nocc,nocc,-1))
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('klid,jd->klij',eris_ooov_klX[iterkk,iterkl,ki],t1[kj])
                    oooo_tmp[iterkk,iterkl,iterki] += einsum('lkjc,ic->klij',eris_ooov_lkX[iterkl,iterkk,kj],t1[ki])
                    Wklij[kl,ki,kj] = oooo_tmp[iterkk,iterkl,iterki]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wklij, op=MPI.SUM)

    return Wklij

def Wvvvv(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wabcd = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir),dtype=t2.dtype)
    else:
        Wabcd = fint['Wvvvv']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    nkpts_blksize2 = 1
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    vvvv_tmp_size = BLKSIZE + (nvir,nvir,nvir,nvir)
    vvvv_tmp = np.empty(vvvv_tmp_size,dtype=t2.dtype)

    print("vvvv blksize")
    print(BLKSIZE)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_vovv = _cp(eris.vovv[s0,s1,s2])
        eris_ovvv = _cp(eris.ovvv[s0,s1,s2])
        eris_oovv_abc = _cp(eris.oovv[s0,s1,s2])

        vvvv_tmp = _cp(eris.vvvv[s0,s1,s2])
        for iterka,ka in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    kd = kconserv[ka,kc,kb]
                    vvvv_tmp[iterka,iterkb,iterkc] += einsum('klcd,ka,lb->abcd',eris_oovv_abc[iterka,iterkb,iterkc],t1[ka],t1[kb])

                    OOvv   = np.empty( (nkpts,nocc,nocc,nvir,nvir), dtype=t2.dtype)
                    t2_tmp = np.empty( (nkpts,nocc,nocc,nvir,nvir), dtype=t2.dtype)
                    #for kk in range(nkpts):
                    #    # kk + kl - kc - kd = 0
                    #    # => kl = kc - kk + kd
                    #    kl = kconserv[kc,kk,kd]
                    #    vvvv_tmp[iterka,iterkb,iterkc] += einsum('klcd,klab->abcd',eris.oovv[kk,kl,kc],t2[kk,kl,ka])
                    for kk in range(nkpts):
                        # kk + kl - kc - kd = 0
                        kl = kconserv[kc,kk,kd]
                        OOvv[kk]   = eris.oovv[kk,kl,kc]
                        #t2_tmp[kk] = t2[kk,kl,ka]
                        t2_tmp[kk] = unpack_tril(t2,nkpts,kk,kl,ka,kconserv[kk,ka,kl])
                    OOvv   = OOvv.reshape(-1,nvir,nvir)
                    t2_tmp = t2_tmp.reshape(-1,nvir,nvir)
                    vvvv_tmp[iterka,iterkb,iterkc] += einsum('xcd,xab->abcd',OOvv,t2_tmp)

                    vvvv_tmp[iterka,iterkb,iterkc] += einsum('alcd,lb->abcd',eris_vovv[iterka,iterkb,iterkc],-t1[kb])
                    #vvvv_tmp[iterka,iterkb,iterkc] += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
                    vvvv_tmp[iterka,iterkb,iterkc] += einsum('kbcd,ka->abcd',eris_ovvv[iterka,iterkb,iterkc],-t1[ka])
        Wabcd[s0,s1,s2] = vvvv_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wabcd, op=MPI.SUM)

    return Wabcd

def Wvvvo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wabcj = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=t2.dtype)
        WWvvvv = Wvvvv(cc,t1,t2,eris)
        WW1ovov = W1ovov(cc,t1,t2,eris)
        WW1ovvo = W1ovvo(cc,t1,t2,eris)
        FFov = cc_Fov(cc,t1,t2,eris)
    else:
        Wabcj = fint['Wvvvo']
        WWvvvv = fint['Wvvvv']
        WW1ovov = fint['W1ovov']
        WW1voov = fint['W1voov']
        FFov = cc_Fov(cc,t1,t2,eris)

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    nkpts_blksize2 = 1
    BLKSIZE = (1,nkpts_blksize2,nkpts_blksize,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    vvvo_tmp_size = BLKSIZE + (nvir,nvir,nvir,nocc)
    vvvo_tmp = np.empty(vvvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_vovv_aXc = _cp(eris.vovv[s0,:,s2])
        eris_ovvv_Xac = _cp(eris.ovvv[:,s0,s2])
        eris_ovvv_Xbc = _cp(eris.ovvv[:,s1,s2])

        Wvvvv_abc = _cp(WWvvvv[s0,s1,s2])
        W1voov_abc = _cp(WW1voov[s1,s0,:])
        W1ovov_baX = _cp(WW1ovov[s1,s0,:])

        for iterka,ka in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    kj = kconserv[ka,kc,kb]
                    vvvo_tmp[iterka,iterkb,iterkc] = np.array(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('abcd,jd->abcj',Wvvvv_abc[iterka,iterkb,iterkc],t1[kj])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lajc,lb->abcj',W1ovov_baX[iterkb,iterka,kj],-t1[kb])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('bkjc,ka->abcj',W1voov_abc[iterkb,iterka,kj],-t1[ka])

                    kl_ranges = range(nkpts)
                    kd_ranges = kconserv[ka,kc,kl_ranges]
                    St2 = np.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t2.dtype)
                    for kl in range(nkpts):
                        # ka + kl - kc - kd = 0
                        # => kd = ka - kc + kl
                        kd = kconserv[ka,kc,kl]
                        #St2[kl] = 2.*t2[kl,kj,kd]
                        St2[kl] = 2.*unpack_tril(t2,nkpts,kl,kj,kd,kconserv[kl,kd,kj])
                        #St2[kl] -= t2[kl,kj,kb].transpose(0,1,3,2)
                        St2[kl] -= unpack_tril(t2,nkpts,kl,kj,kb,kconserv[kl,kb,kj]).transpose(0,1,3,2)
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('alcd,ljdb->abcj',
                                                        eris_vovv_aXc[iterka,:,iterkc].transpose(1,0,2,3,4).reshape(nvir,nkpts*nocc,nvir,nvir),
                                                        St2.reshape(nkpts*nocc,nocc,nvir,nvir))
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lacd,jlbd->abcj',
                                                        eris_ovvv_Xac[:,iterka,iterkc].reshape(nkpts*nocc,nvir,nvir,nvir),
#                                                        -t2[kj,:,kb].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                                                        -unpack_tril(t2,nkpts,kj,range(nkpts),kb,
                                                                kconserv[kj,kb,range(nkpts)]).transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lbcd,ljad->abcj',
                                                        eris_ovvv_Xbc[:,iterkb,iterkc].reshape(nkpts*nocc,nvir,nvir,nvir),
#                                                        -t2[:,kj,ka].reshape(nkpts*nocc,nocc,nvir,nvir))
                                                        -unpack_tril(t2,nkpts,range(nkpts),kj,ka,kconserv[range(nkpts),ka,kj]).reshape(nkpts*nocc,nocc,nvir,nvir))
                    for kl in range(nkpts):
                        kk = kconserv[kb,kl,ka]
                        #vvvo_tmp[iterka,iterkb,iterkc] += einsum('jclk,lkba->abcj',eris.ovoo[kj,kc,kl].conj(),t2[kl,kk,kb])
                        vvvo_tmp[iterka,iterkb,iterkc] += einsum('jclk,lkba->abcj',eris.ovoo[kj,kc,kl].conj(),unpack_tril(t2,nkpts,kl,kk,kb,kconserv[kl,kb,kk]))

                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lc,ljab->abcj',-FFov[kc],unpack_tril(t2,nkpts,kc,kj,ka,kconserv[kc,ka,kj]))

        Wabcj[s0,s1,s2] = vvvo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wabcj, op=MPI.SUM)

    return Wabcj

def WvvvoR1(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wabcj = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=t2.dtype)
        WWvvvv = Wvvvv(cc,t1,t2,eris)
        WW1ovov = W1ovov(cc,t1,t2,eris)
        WW1ovvo = W1ovvo(cc,t1,t2,eris)
        FFov = cc_Fov(cc,t1,t2,eris)
    else:
        Wabcj = fint['WvvvoR1']
        WWvvvv = fint['Wvvvv']
        WW1ovov = fint['W1ovov']
        WW1voov = fint['W1voov']
        FFov = cc_Fov(cc,t1,t2,eris)

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (1,nkpts_blksize2,nkpts_blksize,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    vvvo_tmp_size = BLKSIZE + (nvir,nvir,nvir,nocc)
    vvvo_tmp = np.empty(vvvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_vovv_aXc = _cp(eris.vovv[s0,:,s2])
        eris_ovvv_Xac = _cp(eris.ovvv[:,s0,s2])
        eris_ovvv_Xbc = _cp(eris.ovvv[:,s1,s2])
        eris_vovvR1_cXa = _cp(eris.vovvR1[s0,s2,:])

        Wvvvv_abc = _cp(WWvvvv[s0,s1,s2])
        W1voov_baX = _cp(WW1voov[s1,s0,:])
        W1ovov_baX = _cp(WW1ovov[s1,s0,:])

        for iterka,ka in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    kj = kconserv[ka,kc,kb]
                    #vvvo_tmp[iterka,iterkb,iterkc] = np.array(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
                    vvvo_tmp[iterka,iterkb,iterkc] = np.array(eris_vovvR1_cXa[iterka,iterkc,kj]).transpose(2,3,0,1).conj()
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('abcd,jd->abcj',Wvvvv_abc[iterka,iterkb,iterkc],t1[kj])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lajc,lb->abcj',W1ovov_baX[iterkb,iterka,kj],-t1[kb])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('bkjc,ka->abcj',W1voov_baX[iterkb,iterka,kj],-t1[ka])

                    kl_ranges = range(nkpts)
                    kd_ranges = kconserv[ka,kc,kl_ranges]
                    St2 = np.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t2.dtype)
                    for kl in range(nkpts):
                        # ka + kl - kc - kd = 0
                        # => kd = ka - kc + kl
                        kd = kconserv[ka,kc,kl]
                        #St2[kl] = 2.*t2[kl,kj,kd]
                        St2[kl] = 2.*unpack_tril(t2,nkpts,kl,kj,kd,kconserv[kl,kd,kj])
                        #St2[kl] -= t2[kl,kj,kb].transpose(0,1,3,2)
                        St2[kl] -= unpack_tril(t2,nkpts,kl,kj,kb,kconserv[kl,kb,kj]).transpose(0,1,3,2)
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('alcd,ljdb->abcj',
                                                        eris_vovv_aXc[iterka,:,iterkc].transpose(1,0,2,3,4).reshape(nvir,nkpts*nocc,nvir,nvir),
                                                        St2.reshape(nkpts*nocc,nocc,nvir,nvir))
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('alcd,ljdb->abcj',
                    #                                    eris_vovvR1_aXc[iterkc,iterka,:].transpose(1,0,2,3,4).reshape(nvir,nkpts*nocc,nvir,nvir),
                    #                                    St2.reshape(nkpts*nocc,nocc,nvir,nvir))
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lacd,jlbd->abcj',
                                                        eris_ovvv_Xac[:,iterka,iterkc].reshape(nkpts*nocc,nvir,nvir,nvir),
                                                        -unpack_tril(t2,nkpts,kj,range(nkpts),kb,
                                                            kconserv[kj,kb,range(nkpts)]).transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('lacd,jlbd->abcj',
                    #                                    eris_ovvvRev_Xac[iterkc,iterka,:].reshape(nkpts*nocc,nvir,nvir,nvir),
                    #                                    -t2[kj,:,kb].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lbcd,ljad->abcj',
                                                        eris_ovvv_Xbc[:,iterkb,iterkc].reshape(nkpts*nocc,nvir,nvir,nvir),
                                                        -unpack_tril(t2,nkpts,range(nkpts),kj,ka,
                                                            kconserv[range(nkpts),ka,kj]).reshape(nkpts*nocc,nocc,nvir,nvir))
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('lbcd,ljad->abcj',
                    #                                    eris_ovvvRev_Xbc[iterkc,iterkb,:].reshape(nkpts*nocc,nvir,nvir,nvir),
                    #                                    -t2[:,kj,ka].reshape(nkpts*nocc,nocc,nvir,nvir))
                    #for kl in range(nkpts):
                    #    kk = kconserv[kb,kl,ka]
                    #    vvvo_tmp[iterka,iterkb,iterkc] += einsum('jclk,lkba->abcj',eris.ovoo[kj,kc,kl].conj(),t2[kl,kk,kb])
                    eris_ovoo_jcX = _cp(eris.ovoo[kj,kc,:])
                    t2_tmp = np.empty( (nkpts,nocc,nocc,nvir,nvir), dtype=t2.dtype)
                    for kl in range(nkpts):
                        kk = kconserv[kb,kl,ka]
                        t2_tmp[kl] = unpack_tril(t2,nkpts,kl,kk,kb,kconserv[kl,kb,kk])
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('xjclk,xlkba->abcj',eris.ovoo[kj,kc,kl].conj(),t2[kl,kk,kb])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('jcx,xba->abcj',eris_ovoo_jcX.transpose(1,2,0,3,4).reshape(nocc,nvir,-1).conj(),
                                                            t2_tmp.reshape(-1,nvir,nvir))

                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                    #vvvo_tmp[iterka,iterkb,iterkc] += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
                    vvvo_tmp[iterka,iterkb,iterkc] += einsum('lc,ljab->abcj',-FFov[kc],unpack_tril(t2,nkpts,kc,kj,ka,kconserv[kc,ka,kj]))

        Wabcj[s2,s0,s1] = vvvo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)].transpose(2,0,1,3,4,5,6)
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wabcj, op=MPI.SUM)

    return Wabcj

def Wovoo(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    FFov = cc_Fov(cc,t1,t2,eris)
    if fint is None:
        Wkbij = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=t2.dtype)
        WW1ovov = W1ovov(cc,t1,t2,eris)
        WWoooo = Woooo(cc,t1,t2,eris)
        #WW1ovvo = W1ovvo(cc,t1,t2,eris)
        WW1voov = W1voov(cc,t1,t2,eris)
    else:
        Wkbij = fint['Wovoo']
        WW1ovov = fint['W1ovov']
        WWoooo  = fint['Woooo']
        #WW1ovvo = fint['W1ovvo']
        WW1voov = fint['W1voov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    nkpts_blksize2 = 1
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    print(BLKSIZE)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here 
    ovoo_tmp_size = BLKSIZE + (nocc,nvir,nocc,nocc)
    ovoo_tmp = np.empty(ovoo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        WW1ovov_kbi = _cp(WW1ovov[s0,s1,s2])
        WWoooo_kbi  = _cp(WWoooo[s0,s1,s2])
        #WW1ovvo_kbi = _cp(WW1ovvo[s0,s1,s2])
        WW1voov_bkX = _cp(WW1voov[s1,s0,:])

        eris_vovv_bkX = _cp(eris.vovv[s1,s0,:])
        eris_ooov_XkX = _cp(eris.ooov[:,s0,:])
        eris_ooov_kXi = _cp(eris.ooov[s0,:,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterkb,kb in enumerate(ranges1):
                for iterki,ki in enumerate(ranges2):
                    kj = kconserv[kk,ki,kb]

                    ovoo_tmp[iterkk,iterkb,iterki] = np.array(eris.ovoo[kk,kb,ki],copy=True)
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('kbid,jd->kbij',WW1ovov_kbi[iterkk,iterkb,iterki], t1[kj])
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('klij,lb->kbij', WWoooo_kbi[iterkk,iterkb,iterki],-t1[kb])
                    #ovoo_tmp[iterkk,iterkb,iterki] += einsum('kbcj,ic->kbij',WW1ovvo_kbi[iterkk,iterkb,iterki],t1[ki])
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('bkjc,ic->kbij',WW1voov_bkX[iterkb,iterkk,kj],t1[ki])

                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('lkid,jlbd->kbij', -eris_ooov_XkX[:,iterkk,ki].reshape(nkpts*nocc,nocc,nocc,nvir),
                                                                  unpack_tril(t2,nkpts,kj,range(nkpts),kb,
                                                                      kconserv[kj,kb,range(nkpts)]).transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('lkjd,libd->kbij', -eris_ooov_XkX[:,iterkk,kj].reshape(nkpts*nocc,nocc,nocc,nvir),
                                                                  unpack_tril(t2,nkpts,range(nkpts),ki,kb,
                                                                      kconserv[range(nkpts),kb,ki]).reshape(nkpts*nocc,nocc,nvir,nvir))

                    #St2 = 2.*t2[kj,:,kb]
                    St2 = 2.*unpack_tril(t2,nkpts,kj,range(nkpts),kb,kconserv[kj,kb,range(nkpts)])
                    #St2 -= t2[:,kj,kb].transpose(0,2,1,3,4)
                    St2 -= unpack_tril(t2,nkpts,range(nkpts),kj,kb,kconserv[range(nkpts),kb,kj]).transpose(0,2,1,3,4)
                    St2 = St2.transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir)
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('klid,jlbd->kbij', eris_ooov_kXi[iterkk,:,iterki].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nocc,nvir), St2)

                    #tau1 = t2[kj,ki,:].copy()
                    tau1 = unpack_tril(t2,nkpts,kj,ki,range(nkpts),kconserv[kj,range(nkpts),ki]).copy()
                    tau1[kj] += einsum('jd,ic->jidc',t1[kj],t1[ki])
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('bkdc,jidc->kbij', eris_vovv_bkX[iterkb,iterkk,:].transpose(1,2,0,3,4).reshape(nvir,nocc,nvir*nkpts,nvir),
                                                                tau1.transpose(1,2,0,3,4).reshape(nocc,nocc,nkpts*nvir,nvir))
                    #ovoo_tmp[iterkk,iterkb,iterki] += einsum('kc,ijcb->kbij', FFov[kk],t2[ki,kj,kk])
                    ovoo_tmp[iterkk,iterkb,iterki] += einsum('kc,ijcb->kbij', FFov[kk],unpack_tril(t2,nkpts,ki,kj,kk,kconserv[ki,kk,kj]))

        Wkbij[s0,s1,s2] = ovoo_tmp[:len(ranges0),:len(ranges1),:len(ranges2)]
        loader.slave_finished()

    comm.Barrier()
    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkbij, op=MPI.SUM)

    return Wkbij

def W1voov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)
    else:
        Wkaci  = fint['W1voov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovvo_tmp_size = BLKSIZE + (nocc,nvir,nvir,nocc)
    ovvo_tmp = np.empty(ovvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        eris_ovvo_kac = _cp(eris.ovvo[s0,s1,s2])
        eris_oovv_kXc = _cp(eris.oovv[s0,:,s2])
        eris_oovv_Xkc = _cp(eris.oovv[:,s0,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterka,ka in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    ki = kconserv[kk,kc,ka]
                    ovvo_tmp[iterkk,iterka,iterkc] = _cp(eris_ovvo_kac[iterkk,iterka,iterkc])
                    #St2 = 2.*t2[ki,:,ka]
                    St2 = 2.*unpack_tril(t2,nkpts,ki,range(nkpts),ka,kconserv[ki,ka,range(nkpts)])
                    #St2 -= t2[:,ki,ka].transpose(0,2,1,3,4)
                    St2 -= unpack_tril(t2,nkpts,range(nkpts),ki,ka,kconserv[range(nkpts),ka,ki]).transpose(0,2,1,3,4)
                    ovvo_tmp[iterkk,iterka,iterkc] += einsum('klcd,ilad->kaci',eris_oovv_kXc[iterkk,:,iterkc].transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir),
                                                                St2.transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))
                    ovvo_tmp[iterkk,iterka,iterkc] += -einsum('lkcd,ilad->kaci',eris_oovv_Xkc[:,iterkk,iterkc].reshape(nocc*nkpts,nocc,nvir,nvir),
                                                                unpack_tril(t2,nkpts,ki,range(nkpts),ka,
                                                                    kconserv[ki,ka,range(nkpts)]).transpose(1,0,2,3,4).reshape(nocc,nkpts*nocc,nvir,nvir))

                    Wkaci[ka,kk,ki] = ovvo_tmp[iterkk,iterka,iterkc].transpose(1,0,3,2)

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci

def W2voov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)
        WWooov = Wooov(cc,t1,t2,eris)
    else:
        Wkaci  = fint['W2voov']
        WWooov = fint['Wooov']

    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here
    ovvo_tmp_size = BLKSIZE + (nocc,nvir,nvir,nocc)
    ovvo_tmp = np.empty(ovvo_tmp_size,dtype=t2.dtype)

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        Wooov_akX     = _cp(WWooov[s1,s0])
        eris_ovvv_kac = _cp(eris.ovvv[s0,s1,s2])

        for iterkk,kk in enumerate(ranges0):
            for iterka,ka in enumerate(ranges1):
                for iterkc,kc in enumerate(ranges2):
                    ki = kconserv[kk,kc,ka]
                    ovvo_tmp[iterkk,iterka,iterkc] = einsum('la,lkic->kaci',-t1[ka],Wooov_akX[iterka,iterkk,ki])
                    ovvo_tmp[iterkk,iterka,iterkc] += einsum('akdc,id->kaci',eris_ovvv_kac[iterkk,iterka,iterkc].transpose(1,0,3,2),t1[ki])

                    Wkaci[ka,kk,ki] = ovvo_tmp[iterkk,iterka,iterkc].transpose(1,0,3,2)

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci


def Wvoov(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)
        W1kaci = W1voov(cc,t1,t2,eris,fint)
        W2kaci = W2voov(cc,t1,t2,eris,fint)
    else:
        Wkaci = fint['Wvoov']
        W1kaci = fint['W1voov']
        W2kaci = fint['W2voov']

    # TODO can do much better than this... call recursive function
    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        Wkaci[s0,s1,s2] = _cp(W1kaci[s0,s1,s2]) + _cp(W2kaci[s0,s1,s2])

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci

def WvoovR1(cc,t1,t2,eris,fint=None):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv

    if fint is None:
        Wkaci  = np.zeros((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir),dtype=t1.dtype)
        W1kaci = W1voov(cc,t1,t2,eris,fint)
        W2kaci = W2voov(cc,t1,t2,eris,fint)
    else:
        Wkaci = fint['WvoovR1']
        W1kaci = fint['W1voov']
        W2kaci = fint['W2voov']

    # TODO can do much better than this... call recursive function
    # Adaptive blocking begins here
    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(np.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(np.floor(mem/(pre*nkpts_blksize))),1),nkpts)
    BLKSIZE = (nkpts_blksize2,nkpts_blksize,nkpts,)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))
    # Adaptive blocking ends here

    good2go = True
    while(good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        Wkaci[s2,s0,s1] = (_cp(W1kaci[s0,s1,s2]) + _cp(W2kaci[s0,s1,s2])).transpose(2,0,1,3,4,5,6)

        loader.slave_finished()

    comm.Barrier()

    if fint is None:
        comm.Allreduce(MPI.IN_PLACE, Wkaci, op=MPI.SUM)

    return Wkaci

def _cp(a):
    return np.array(a, copy=False, order='C')
