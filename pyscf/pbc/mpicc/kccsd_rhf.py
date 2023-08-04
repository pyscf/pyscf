#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import copy
from functools import reduce

import numpy
import os
import numpy as np
import h5py

from pyscf.pbc.mpicc import mpi_kpoint_helper
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.pbc.mp.kmp2 import get_frozen_mask, get_nocc, get_nmo, padded_mo_coeff, padding_k_idx
from pyscf.pbc.mpicc import kintermediates_rhf as imdk
from pyscf.pbc.lib.linalg_helper import eigs
from pyscf.lib.linalg_helper import eig
from pyscf.pbc.mpitools.mpi_helper import generate_max_task_list, safeAllreduceInPlace, safeNormDiff, safeBcastInPlace
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.pbc.mpitools import mpi_load_balancer, mpi
from pyscf.pbc.tools.tril import tril_index, unpack_tril
from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.cc.kccsd_rhf
from pyscf.pbc.cc.eom_kccsd_ghf import mask_frozen_ea
from pyscf.pbc.cc.eom_kccsd_ghf import mask_frozen_ip

from mpi4py import MPI

#einsum = np.einsum
einsum = lib.einsum

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def read_amplitudes(t1_shape, t2_shape, t1=None, t2=None, filename="t_amplitudes.hdf5"):
    task_list = generate_max_task_list(t2_shape)
    read_success = False
    if os.path.isfile(filename):
        if t1 is None:
            t1 = np.empty(t1_shape)
        if t2 is None:
            t2 = np.empty(t2_shape)
        print("reading t amplitudes from file")
        feri = h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        saved_t1 = feri['t1']
        saved_t2 = feri['t2']
        assert (saved_t1.shape == t1_shape)
        assert (saved_t2.shape == t2_shape)

        task_list = generate_max_task_list(t1.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            t1[tuple(which_slice)] = saved_t1[tuple(which_slice)]
        task_list = generate_max_task_list(t2.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            t2[tuple(which_slice)] = saved_t2[tuple(which_slice)]
        feri.close()
        read_success = True
    return read_success, t1, t2

def write_amplitudes(t1, t2, filename="t_amplitudes.hdf5"):
    task_list = generate_max_task_list(t2.shape)
    if rank == 0:
        print("writing t amplitudes to file")
        feri = h5py.File(filename, 'w')
        ds_type = t2.dtype
        out_t1  = feri.create_dataset('t1', t1.shape, dtype=ds_type)
        out_t2  = feri.create_dataset('t2', t2.shape, dtype=ds_type)

        task_list = generate_max_task_list(t1.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            out_t1[tuple(which_slice)] = t1[tuple(which_slice)]
        task_list = generate_max_task_list(t2.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            out_t2[tuple(which_slice)] = t2[tuple(which_slice)]
        feri.close()
    return

def read_eom_amplitudes(vec_shape, filename="reom_amplitudes.hdf5", vec=None):
    task_list = generate_max_task_list(vec_shape)
    read_success = False

    # TODO: find a way to make the amplitudes are consistent
    # with the signs of the eris/t-amplitudes when restarting
    return False, None

    print("attempting to read in eom amplitudes from file ", filename)
    if os.path.isfile(filename):
        print("reading eom amplitudes from file. shape=", vec_shape)
        feri = h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        saved_v = feri['v']
        if vec is None:
            vec = np.empty(vec_shape,dtype=saved_v.dtype)
        assert (saved_v.shape == vec_shape)
        task_list = generate_max_task_list(vec.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            vec[tuple(which_slice)] = saved_v[tuple(which_slice)]
        feri.close()
        read_success = True
    if vec is not None and vec_shape[-1] == 1:
        vec = vec.reshape(vec_shape[:-1])
    return read_success, vec

def write_eom_amplitudes(vec, filename="reom_amplitudes.hdf5"):
    task_list = generate_max_task_list(vec.shape)
    if rank == 0:
        print("writing eom amplitudes to file")
        feri = h5py.File(filename, 'w')
        ds_type = vec.dtype
        out_v  = feri.create_dataset('v', vec.shape, dtype=ds_type)

        task_list = generate_max_task_list(vec.shape)
        for block in task_list:
            which_slice = [slice(*x) for x in block]
            out_v[tuple(which_slice)] = vec[tuple(which_slice)]
        feri.close()
    return

def restore_from_diis_(mycc, diis_file, inplace=True):
    '''Reuse an existed DIIS object in the CCSD calculation.
    The CCSD amplitudes will be restored from the DIIS object to generate t1
    and t2 amplitudes. The t1/t2 amplitudes of the CCSD object will be
    overwritten by the generated t1 and t2 amplitudes. The amplitudes vector
    and error vector will be reused in the CCSD calculation.
    '''
    adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
    if rank == 0:
        adiis.restore(diis_file, inplace=inplace)

        ccvec = adiis.extrapolate()
        t1, t2 = mycc.vector_to_amplitudes(ccvec)
    info = None
    if rank == 0:
        info = (t1.shape, t2.shape, np.result_type(t1, t2))
    info = MPI.COMM_WORLD.bcast(info)

    if rank != 0:  # Create empty arrays for master to bcast into
        t1_shape, t2_shape, dtype = info
        t1 = np.empty(t1_shape, dtype=dtype)
        t2 = np.empty(t2_shape, dtype=dtype)
    safeBcastInPlace(MPI.COMM_WORLD, t1)
    safeBcastInPlace(MPI.COMM_WORLD, t2)
    mycc.t1, mycc.t2 = t1, t2
    if inplace:
        mycc.diis = adiis
    return mycc

def get_normt_diff(cc, t1, t2, t1new, t2new):
    '''Calculates norm(t1 - t1new) + norm(t2 - t2new).'''
    normt = safeNormDiff(t1new, t1) + safeNormDiff(t2new, t2)  # Blocking; saves memory
    return normt

def update_t1(cc,t1,t2,eris,ints1e):
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    fov = fock[:,:nocc,nocc:]

    Foo,Fvv,Fov,Loo,Lvv = ints1e

    kconserv = cc.kconserv
    t1new = numpy.zeros((nkpts,nocc,nvir),dtype=t1.dtype)

    mem = 0.5e9
    pre = 1.*nocc*nocc*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(numpy.sqrt(int(numpy.floor(mem/pre))))),1),nkpts)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,))
    loader.set_ranges((range(nkpts),))

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0 = loader.get_blocks_from_data(data)

        s0 = slice(min(ranges0),max(ranges0)+1)

        eris_voov_aXi = _cp(eris.voov[s0,:,s0])
        eris_ovov_Xai = _cp(eris.ovov[:,s0,s0])

        for iterka,ka in enumerate(ranges0):
            ki = ka
            iterki = iterka
            # kc == ki; kk == ka
            t1new[ka] = fov[ka].conj().copy()
            t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
            t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
            t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])

            tau_term = numpy.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
            for kk in range(nkpts):
                #tau_term[kk] = 2*t2[kk,ki,kk]
                tau_term[kk] = 2*unpack_tril(t2,nkpts,kk,ki,kk,kconserv[kk,kk,ki])
                #tau_term[kk] -= t2[ki,kk,kk].transpose(1,0,2,3)
                tau_term[kk] -= unpack_tril(t2,nkpts,ki,kk,kk,kconserv[ki,kk,kk]).transpose(1,0,2,3)
            tau_term[ka] += einsum('ic,ka->kica',t1[ki],t1[ka])

            t1new[ka] += einsum('kc,kica->ia',
                                Fov[:].reshape(nocc*nkpts,nvir),
                                tau_term[:].reshape(nocc*nkpts,nocc,nvir,nvir))

            t1new[ka] += einsum('akic,kc->ia',
                                eris_voov_aXi[iterka,:,iterki].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                2*t1[:].reshape(nocc*nkpts,nvir))
            t1new[ka] += einsum('kaic,kc->ia',
                                eris_ovov_Xai[:,iterka,iterki].reshape(nocc*nkpts,nvir,nocc,nvir),
                                -t1[:].reshape(nocc*nkpts,nvir))
        loader.slave_finished()

    comm.Barrier()

    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,nkpts_blksize2,))
    loader.set_ranges((range(nkpts),range(nkpts),))

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0,ranges1 = loader.get_blocks_from_data(data)

        s0,s1= [slice(min(x),max(x)+1) for x in (ranges0,ranges1)]

        eris_ovvv_kaX = _cp(eris.ovvv[s1,s0,:])
        eris_ooov_kXi = _cp(eris.ooov[s1,:,s0])
        eris_ooov_Xki = _cp(eris.ooov[:,s1,s0])

        for iterka,ka in enumerate(ranges0):
            ki = ka
            iterki = iterka
            for iterkk,kk in enumerate(ranges1):
                kd_list = _cp(kconserv[ka,range(nkpts),kk])
                kc_list = _cp(range(nkpts))
                Svovv = (2*eris_ovvv_kaX[iterkk,iterka,kd_list].transpose(0,2,1,4,3)
                         - eris_ovvv_kaX[iterkk,iterka,kc_list].transpose(0,2,1,3,4))
                #tau_term_1 = t2[ki,kk,:].copy()
                tau_term_1 = unpack_tril(t2,nkpts,ki,kk,range(nkpts),kconserv[ki,range(nkpts),kk]).copy()
                tau_term_1[ki] += einsum('ic,kd->ikcd',t1[ki],t1[kk])
                t1new[ka] += einsum('ak,ik->ia',Svovv.transpose(1,2,0,3,4).reshape(nvir,-1),
                                                tau_term_1.transpose(1,2,0,3,4).reshape(nocc,-1))

                kl_list = _cp(kconserv[ki,kk,range(nkpts)])
                Sooov = (2*eris_ooov_kXi[iterkk,kl_list,iterki]
                         - eris_ooov_Xki[kl_list,iterkk,iterki].transpose(0,2,1,3,4))
                #tau_term_1 = t2[kk,kl_list,ka].copy()
                tau_term_1 = unpack_tril(t2,nkpts,kk,kl_list,ka,kconserv[kk,ka,kl_list]).copy()
                if kk == ka:
                    tau_term_1[kc_list==kl_list] += einsum('ka,xlc->xklac',t1[ka],t1[kc_list==kl_list])
                t1new[ka] += -einsum('ki,ka->ia',Sooov.transpose(0,1,2,4,3).reshape(-1,nocc),
                                     tau_term_1.transpose(0,1,2,4,3).reshape(-1,nvir))

        loader.slave_finished()

    comm.Allreduce(MPI.IN_PLACE, t1new, op=MPI.SUM)
    return t1new

# This is restricted (R)CCSD
# following Hirata, ..., Barlett, J. Chem. Phys. 120, 2581 (2004)

def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    tril_shape = ((nkpts)*(nkpts+1))//2

    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
    #t2tmp = numpy.zeros((tril_shape,nkpts,nocc,nocc,nvir,nvir),dtype=t2.dtype)
    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        for ka in range(nkpts):
    #            if ki <= kj:
    #                t2tmp[tril_index(ki,kj),ka] = t2[ki,kj,ka]
    #t2 = t2tmp

    #mo_e = eris.fock.diagonal()
    #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    ds_type = t1.dtype

    #Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    #Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    #Wvoov = imdk.cc_Wvoov(cc,t1,t2,eris)
    #Wvovo = imdk.cc_Wvovo(cc,t1,t2,eris)

    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Loo = imdk.Loo(cc,t1,t2,eris)
    Lvv = imdk.Lvv(cc,t1,t2,eris)

    if rank == 0:
        print("done making intermediates...")
    # Move energy terms to the other side
    for k in range(nkpts):
        Foo[k][np.diag_indices(nocc)] -= mo_e_o[k]
        Fvv[k][np.diag_indices(nvir)] -= mo_e_v[k]
        Loo[k][np.diag_indices(nocc)] -= mo_e_o[k]
        Lvv[k][np.diag_indices(nvir)] -= mo_e_v[k]

    kconserv = cc.kconserv

    if rank == 0:
        print("t1 equation")
    # T1 equation
    t1new = update_t1(cc,t1,t2,eris,[Foo,Fvv,Fov,Loo,Lvv])

    if rank == 0:
        print("t2 equation")
    # T2 equation
    #t2new = numpy.array(eris.oovv, copy=True).conj()
    #t2new = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir),dtype=ds_type)
    t2new_tril = numpy.zeros((tril_shape,nkpts,nocc,nocc,nvir,nvir),dtype=ds_type)

    cput2 = logger.process_clock(), logger.perf_counter()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,nkpts,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    # Figuring out number of kpoints we can have in our oovv terms below
    # TODO : clean this up- just temporary
    mem = 0.5e9
    pre = 1.*nkpts*nkpts*nocc*nocc*nvir*nvir*16
    nkpts_blksize = max(int(numpy.floor(mem/pre)),1)
    BLKSIZE2 = min(nkpts,nkpts_blksize)
    BLKSIZE2_ranges = [(BLKSIZE2*i,min(nkpts,BLKSIZE2*(i+1)))
                       for i in range(int(numpy.ceil(1.*nkpts/BLKSIZE2)))]

    # Making Woooo terms
    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        eris_oovv = _cp(eris.oovv[s0,s1,s2])

        eris_oooo = _cp(eris.oooo[s1,s0])
        eris_ovoo_ij = _cp(eris.ovoo[s0,s1])
        eris_ovoo_ji = _cp(eris.ovoo[s1,s0])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    #t2new[ki,kj,ranges2] += _cp(eris_oovv[iterki,iterkj,s2]).conj()
                    t2new_tril[tril_index(ki,kj),ranges2] += _cp(eris_oovv[iterki,iterkj,s2]).conj()

        for kblock in BLKSIZE2_ranges:
            kl_block_size = kblock[1]-kblock[0]
            # Find out how large of a block_size we need
            kklist = []
            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        kk = kconserv[kj,kl,ki]
                        kklist.append(kk)
                        iterkk = numpy.where(numpy.asarray(kklist)==kk)[0]
                        if len(iterkk)==0: #if not found, append
                            kklist.append(kk)

            kk_block_size = len(kklist)
            eris_oovv1 = numpy.empty((kk_block_size,kl_block_size,nkpts,nocc,nocc,nvir,nvir),
                                     dtype=t2.dtype)
            #  Now fill in the matrix elements
            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        kk = kconserv[kj,kl,ki]
                        iterkk = numpy.where(numpy.asarray(kklist)==kk)[0][0]
                        eris_oovv1[iterkk,iterkl,:] = _cp(eris.oovv[kk,kl,:])

            for iterkl,kl in enumerate(range(kblock[0],kblock[1])):
                for iterki,ki in enumerate(ranges0):
                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            kk = kconserv[kj,kl,ki]
                            iterkk = numpy.where(kklist==kk)[0][0]

                            #wOOoo = numpy.empty((nkpts,nocc,nocc,nocc,nocc),dtype=t2.dtype)
                            #tau1_ooVV = t2[ki,kj,:].copy()
                            tau1_ooVV = unpack_tril(t2,nkpts,ki,kj,range(nkpts),kconserv[ki,range(nkpts),kj])
                            tau1_ooVV[ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])

                            # TODO read only packed oovv terms and unpack after reading
                            wOOoo = _cp(eris_oooo[iterkj,iterki,kl].transpose(3,2,1,0)).conj()
                            wOOoo += einsum('klic,jc->klij',eris_ovoo_ij[iterki,iterkj,kk].transpose(2,3,0,1).conj(),t1[kj])
                            wOOoo += einsum('lkjc,ic->klij',eris_ovoo_ji[iterkj,iterki,kl].transpose(2,3,0,1).conj(),t1[ki])
                            wOOoo += einsum('klcd,ijcd->klij',
                                            eris_oovv1[iterkk,iterkl,:].transpose(1,2,0,3,4).reshape(nocc,nocc,nkpts*nvir,nvir),
                                            tau1_ooVV.transpose(1,2,0,3,4).reshape(nocc,nocc,nkpts*nvir,nvir))

                            for iterka,ka in enumerate(ranges2):
                                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                                kb = kconserv[ki,ka,kj]
                                kn = kconserv[kj,kl,ki]
                                #tau1_OOvv = t2[kn,kl,ka].copy()
                                tau1_OOvv = unpack_tril(t2,nkpts,kn,kl,ka,kconserv[kn,ka,kl])
                                if ka == kk and kl == kb:
                                    tau1_OOvv += einsum('ka,lb->klab',t1[ka],t1[kb])
                                tmp = einsum('klij,klab->ijab',wOOoo,tau1_OOvv) #kl combined into one
                                t2new_tril[tril_index(ki,kj),ka] += tmp
                                #t2new[ki,kj,ka] += tmp

        loader.slave_finished()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Woooo', *cput2)

    cput2 = logger.process_clock(), logger.perf_counter()

    mem = 0.5e9
    pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    if rank == 0:
        print("vvvv blocksize = ", nkpts_blksize)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    # Making Wvvvv terms. Notice the change of for loops.
    def func3():
        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
            if min(ranges1) > max(ranges2): #continue if ka > kb
                loader.slave_finished()
                continue

            s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

            eris_ovvv_ab = _cp(eris.ovvv[s1,s2])
            eris_vovv_ab = _cp(eris.vovv[s1,s2])
            eris_vvvv_ab = _cp(eris.vvvv[s1,s2])

            for iterka,ka in enumerate(ranges1):
                for iterkb,kb in enumerate(ranges2):
                    if ka <= kb:
                        # Wvvvv term
                        ovVV = eris_ovvv_ab[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nocc,nvir,-1)
                        voVV = eris_vovv_ab[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nvir,nocc,-1)
                        wvvVV = einsum('akd,kb->abd',voVV,-t1[kb])
                        wvvVV += einsum('ak,kbd->abd',-t1[ka].T,ovVV)
                        wvvVV += eris_vvvv_ab[iterka,iterkb].transpose(1,2,0,3,4).reshape(nvir,nvir,-1)
                        wvvVV = wvvVV.transpose(2,0,1)

                        kj_list = kconserv[kb,ranges0,ka]
                        tau1_ooVV = numpy.zeros((len(ranges0),nkpts,nocc,nocc,nvir,nvir),dtype=t2.dtype)
                        for iterki,ki in enumerate(ranges0):
                            kj = kj_list[iterki]
                            #tau1_ooVV[iterki]    += t2[ki,kj,:]
                            tau1_ooVV[iterki] += unpack_tril(t2,nkpts,ki,kj,range(nkpts),kconserv[ki,range(nkpts),kj])
                            tau1_ooVV[iterki,ki] += einsum('ic,jd->ijcd',t1[ki],t1[kj])
                        tau1_ooVV = tau1_ooVV.transpose(0,2,3,1,4,5).reshape(len(ranges0),nocc,nocc,-1)
                        tmp = einsum('kijd,dab->kijab',tau1_ooVV,wvvVV)


                        for iterki,ki in enumerate(ranges0):
                            kj = kj_list[iterki]
                            if ki == kj:
                                t2new_tril[tril_index(ki,kj),ka] += tmp[iterki]
                                #t2new[ki,kj,ka] += tmp[iterki]
                                if ka < kb:
                                    t2new_tril[tril_index(kj,ki),kb] += tmp[iterki].transpose(1,0,3,2)
                                    #t2new[kj,ki,kb] += tmp[iterki].transpose(1,0,3,2)
                            elif ki < kj:
                                t2new_tril[tril_index(ki,kj),ka] += tmp[iterki]
                                #t2new[ki,kj,ka] += tmp[iterki]
                            elif ki > kj:
                                if ka < kb:
                                    t2new_tril[tril_index(kj,ki),kb] += tmp[iterki].transpose(1,0,3,2)
                                    #t2new[kj,ki,kb] += tmp[iterki].transpose(1,0,3,2)
            loader.slave_finished()
    func3()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvvvv', *cput2)

    # Making Wvoov and Wovov terms. (part 1/2)
    cput2 = logger.process_clock(), logger.perf_counter()

    mem = 0.5e9
    pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
    nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]
        # TODO this can sometimes not be optimal for ooov, calls for all kb,
        # but in most block set-ups you only need 1 index
        eris_ooov_ji = _cp(eris.ooov[s1,s0])

        eris_voovR1_aXi = _cp(eris.voovR1[s0,s2,:])
        eris_ooovR1_aXi = _cp(eris.ooovR1[s0,s2,:])
        eris_vovvR1_aXi = _cp(eris.vovvR1[s0,s2,:])

        eris_ovovRev_Xai = _cp(eris.ovovRev[s0,s2,:])
        eris_ooovRev_Xai = _cp(eris.ooovRev[s0,s2,:])
        eris_ovvvRev_Xai = _cp(eris.ovvvRev[s0,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    for iterka,ka in enumerate(ranges2):
                        kb = kconserv[ki,ka,kj]
                        # t2 with 1-electron terms (1/2)
                        #tmp = einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
                        tmp = einsum('ac,ijcb->ijab',Lvv[ka],unpack_tril(t2,nkpts,ki,kj,ka,kconserv[ki,ka,kj]))
                        #tmp += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
                        tmp += einsum('ki,kjab->ijab',-Loo[ki],unpack_tril(t2,nkpts,ki,kj,ka,kconserv[ki,ka,kj]))
                        # t1 with ooov terms       (1/2)
                        tmp2 = (eris_ooov_ji[iterkj,iterki,kb].transpose(3,2,1,0).conj() +
                                einsum('akic,jc->akij',eris_voovR1_aXi[iterki,iterka,kb],t1[kj])) #ooov[kj,ki,kb,ka] ovvo[kb,ka,kj,ki]
                        tmp -= einsum('akij,kb->ijab',tmp2,t1[kb])
                        if ki == kj:
                            t2new_tril[tril_index(ki,kj),ka] += tmp
                            t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                            #t2new[ki,kj,ka] += tmp
                            #t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                        else:
                            t2new_tril[tril_index(ki,kj),ka] += tmp
                            #t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            oOvV   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterki,ki in enumerate(ranges0):
                for iterka,ka in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,ki,ka]
                    # Wvoov term (ka,kk,ki,kc)
                    #    a) the Soovv and oovv contribution to Wvoov is done after the Wovov term, where
                    #        Soovv = 2*oovv[l,k,c,d] - oovv[l,k,d,c]
                    _WvOoV  = _cp(eris_voovR1_aXi[iterki,iterka,kk_slice]).transpose(1,3,0,2,4).reshape(nvir,nocc,-1)                               #voov[ka,*,ki,*]
                    _WvOoV -= einsum('lic,la->aic',eris_ooovR1_aXi[iterki,iterka,kk_slice].transpose(1,3,0,2,4).reshape(nocc,nocc,-1),t1[ka])       #ooov[ka,*,ki,*]
                    _WvOoV += einsum('adc,id->aic',eris_vovvR1_aXi[iterki,iterka,kk_slice].transpose(1,3,0,2,4).reshape(nvir,nvir,-1),t1[ki])       #vovv[ka,*,ki,*]
                    # Wovov term (kk,ka,ki,kc)
                    _WOvoV = _cp(eris_ovovRev_Xai[iterki,iterka,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,ka,ki,*]
                    _WOvoV -= einsum('lic,la->aic',eris_ooovRev_Xai[iterki,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[ka]) #ooov[*,ka,ki,*]
                    _WOvoV += einsum('adc,id->aic',eris_ovvvRev_Xai[iterki,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[ki]) #ovvv[*,ka,ki,*]
                    # Filling in the oovv terms
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                        oOvV[:,iterkk] = _cp(eris.oovv[kk,:,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)
                    oOvV_f = oOvV.transpose(0,3,5,1,2,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    #tau2_OovV  = t2[:,ki,ka].copy()
                    tau2_OovV  = unpack_tril(t2,nkpts,range(nkpts),ki,ka,kconserv[range(nkpts),ka,ki])
                    tau2_OovV[ka] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                    tau2_OovV = tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1)

                    _WvOoV -= 0.5*einsum('dc,iad->aic',oOvV_f,tau2_OovV) # kc consolidated into c, ld consolidated into d
                    _WOvoV -= 0.5*einsum('dc,iad->aic',oOVv_f,tau2_OovV)
                    #_WvOoV += 0.5*einsum('dc,iad->aic',2*oOvV_f-oOVv_f,t2[ki,:,ka].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                    _WvOoV += 0.5*einsum('dc,iad->aic', 2*oOvV_f-oOVv_f,
                                          unpack_tril(t2,nkpts,ki,range(nkpts),ka,kconserv[ki,ka,range(nkpts)]).transpose(1,3,0,2,4).reshape(nocc,nvir,-1))

                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            kb = kconserv[ki,ka,kj]
                            #tmp = einsum('aic,jbc->ijab',(2*_WvOoV-_WOvoV),t2[kj,kk_slice,kb].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            tmp = einsum('aic,jbc->ijab',2*_WvOoV-_WOvoV,
                                         unpack_tril(t2,nkpts,kj,kk_range,kb,kconserv[kj,kb,kk_range]).transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            #tmp -= einsum('aic,jbc->ijab',_WvOoV,t2[kk_slice,kj,kb].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            tmp -= einsum('aic,jbc->ijab',_WvOoV,
                                          unpack_tril(t2,nkpts,kk_range,kj,kb,kconserv[kk_range,kb,kj]).transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            if ki == kj:
                                t2new_tril[tril_index(ki,kj),ka] += tmp
                                t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                                #t2new[ki,kj,ka] += tmp
                                #t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                            else:
                                t2new_tril[tril_index(ki,kj),ka] += tmp
                                #t2new[ki,kj,ka] += tmp
                    #kj_ranges = ranges1[ranges1 >= ki]
                    #nkj = kj_ranges.shape[0]
                    #kb_ranges = kconserv[ki,ka,kj_ranges]
                    #t2new[ki,kj_ranges,ka] += einsum('aic,xjbc->xijab',
                    #                                 (2*_WvOoV-_WOvoV),
                    #                                 t2[kj_ranges,kk_slice,kb_ranges].transpose(0,2,4,1,3,5).reshape(nkj,nocc,nvir,-1))
                    #t2new[ki,kj_ranges,ka] -= einsum('aic,xjbc->xijab',_WvOoV,t2[kk_slice,kj_ranges,kb_ranges].transpose(1,3,4,0,2,5).reshape(nkj,nocc,nvir,-1))
        loader.slave_finished()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvoov (ai)', *cput2)

    # Making Wvoov and Wovov terms (part 2/2)

    cput2 = logger.process_clock(), logger.perf_counter()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) >= max(ranges1): #continue if ki >= kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        # TODO this is not optimal for ooov, calls for all ka, but in most block set-ups you only need 1 index
        eris_ooov_ij = _cp(eris.ooov[:,s1])

        eris_voovR1_bXj = _cp(eris.voovR1[s1,s2,:])
        eris_ooovR1_bXj = _cp(eris.ooovR1[s1,s2,:])
        eris_vovvR1_bXj = _cp(eris.vovvR1[s1,s2,:])

        eris_ovovRev_Xbj = _cp(eris.ovovRev[s1,s2,:])
        eris_ooovRev_Xbj = _cp(eris.ooovRev[s1,s2,:])
        eris_ovvvRev_Xbj = _cp(eris.ovvvRev[s1,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki < kj:
                    for iterkb,kb in enumerate(ranges2):
                        ka = kconserv[ki,kb,kj]
                        # t2 with 1-electron terms (2/2)
                        #tmp = einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])
                        tmp = einsum('bc,jica->ijab',Lvv[kb],unpack_tril(t2,nkpts,kj,ki,kb,kconserv[kj,kb,ki]))
                        #tmp += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])
                        tmp += einsum('kj,kiba->ijab',-Loo[kj],unpack_tril(t2,nkpts,kj,ki,kb,kconserv[kj,kb,ki]))
                        # t1 with ooov terms (2/2)
                        tmp2 = (eris_ooov_ij[iterki,iterkj,ka].transpose(3,2,1,0).conj() +
                                einsum('bkjc,ic->bkji',eris_voovR1_bXj[iterkj,iterkb,ka],t1[ki])) #ooov[ki,kj,ka,kb] ovvo[ka,kb,ki,kj]
                        tmp -= einsum('bkji,ka->ijab',tmp2,t1[ka])
                        t2new_tril[tril_index(ki,kj),ka] += tmp
                        #t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            oOvV   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterkj,kj in enumerate(ranges1):
                for iterkb,kb in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,kj,kb]
                    # Wvoov term (kb,kk,kj,kc)
                    _WvOoV  = _cp(eris_voovR1_bXj[iterkj,iterkb,kk_slice]).transpose(1,3,0,2,4).reshape(nvir,nocc,-1)                          #voov[kb,*,kj,*]
                    _WvOoV -= einsum('ljc,lb->bjc',eris_ooovR1_bXj[iterkj,iterkb,kk_slice].transpose(1,3,0,2,4).reshape(nocc,nocc,-1),t1[kb])  #ooov[kb,*,kj,*]
                    _WvOoV += einsum('bdc,jd->bjc',eris_vovvR1_bXj[iterkj,iterkb,kk_slice].transpose(1,3,0,2,4).reshape(nvir,nvir,-1),t1[kj])  #vovv[kb,*,kj,*]
                    # Wovov term (kk,kb,kj,kc)
                    _WOvoV = _cp(eris_ovovRev_Xbj[iterkj,iterkb,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,kb,kj,*]
                    _WOvoV -= einsum('ljc,lb->bjc',eris_ooovRev_Xbj[iterkj,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[kb]) #ooov[*,kb,kj,*]
                    _WOvoV += einsum('bdc,jd->bjc',eris_ovvvRev_Xbj[iterkj,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[kj]) #ovvv[*,kb,kj,*]
                    #
                    # Filling in the oovv terms
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                        oOvV[:,iterkk] = _cp(eris.oovv[kk,:,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)
                    oOvV_f = oOvV.transpose(0,3,5,1,2,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    #tau2_OovV  = t2[:,kj,kb].copy()
                    tau2_OovV  = unpack_tril(t2,nkpts,range(nkpts),kj,kb,kconserv[range(nkpts),kb,kj])
                    tau2_OovV[kb] += 2*einsum('jd,lb->ljbd',t1[kj],t1[kb])
                    tau2_OovV = tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1)

                    _WvOoV -= 0.5*einsum('dc,jbd->bjc',oOvV_f,tau2_OovV) # kc consolidated into c, ld consolidated into d
                    _WOvoV -= 0.5*einsum('dc,jbd->bjc',oOVv_f,tau2_OovV)
                    #_WvOoV += 0.5*einsum('dc,jbd->bjc',2*oOvV_f-oOVv_f,t2[kj,:,kb].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                    _WvOoV += 0.5*einsum('dc,jbd->bjc',2*oOvV_f-oOVv_f,
                                         unpack_tril(t2,nkpts,kj,range(nkpts),kb,kconserv[kj,kb,range(nkpts)]).transpose(1,3,0,2,4).reshape(nocc,nvir,-1))

                    for iterki,ki in enumerate(ranges0):
                        if ki < kj:
                            ka = kconserv[ki,kb,kj]
                            #tmp = einsum('bjc,iac->ijab',(2*_WvOoV-_WOvoV),t2[ki,kk_slice,ka].transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            tmp = einsum('bjc,iac->ijab',2*_WvOoV-_WOvoV,
                                         unpack_tril(t2,nkpts,ki,kk_range,ka,kconserv[ki,ka,kk_range]).transpose(1,3,0,2,4).reshape(nocc,nvir,-1))
                            #tmp -= einsum('bjc,iac->ijab',_WvOoV,t2[kk_slice,ki,ka].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            tmp -= einsum('bjc,iac->ijab',_WvOoV,
                                          unpack_tril(t2,nkpts,kk_range,ki,ka,kconserv[kk_range,ka,ki]).transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            t2new_tril[tril_index(ki,kj),ka] += tmp
                            #t2new[ki,kj,ka] += tmp
        loader.slave_finished()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Wvoov (bj)', *cput2)

    cput2 = logger.process_clock(), logger.perf_counter()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,nkpts,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    # Making last of the Wovov terms (part 1/2)

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) > max(ranges1): #continue if ki > kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        eris_ovovRev_Xbi = _cp(eris.ovovRev[s0,s2,:])
        eris_ooovRev_Xbi = _cp(eris.ooovRev[s0,s2,:])
        eris_ovvvRev_Xbi = _cp(eris.ovvvRev[s0,s2,:])

        eris_vovvL1_jib = _cp(eris.vovvL1[s0,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki <= kj:
                    for iterkb,kb in enumerate(ranges2):
                        ka = kconserv[ki,kb,kj]
                        # t1 with ovvv terms  (part 1/2)
                        tmp2 = (eris_vovvL1_jib[iterki,iterkb,iterkj].transpose(3,2,1,0).conj() -
                                einsum('kbic,ka->abic',eris_ovovRev_Xbi[iterki,iterkb,ka],t1[ka])) #ovvv[ki,kj,ka,kb]  ovov[ka,kb,ki,kj]
                        tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
                        if ki == kj:
                            t2new_tril[tril_index(ki,kj),ka] += tmp
                            t2new_tril[tril_index(ki,kj),kb] += tmp.transpose(1,0,3,2)
                            #t2new[ki,kj,ka] += tmp
                            #t2new[ki,kj,kb] += tmp.transpose(1,0,3,2)
                        else:
                            t2new_tril[tril_index(ki,kj),ka] += tmp
                            #t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterki,ki in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,ki,kb]
                    # Wovov term (kk,kb,ki,kc)
                    _WOvoV = _cp(eris_ovovRev_Xbi[iterki,iterkb,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,kb,ki,*]
                    _WOvoV -= einsum('lic,lb->bic',eris_ooovRev_Xbi[iterki,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[kb]) #ooov[*,kb,ki,*]
                    _WOvoV += einsum('bdc,id->bic',eris_ovvvRev_Xbi[iterki,iterkb,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[ki]) #ovvv[*,kb,ki,*]
                    #
                    # Filling in the oovv terms
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    #tau2_OovV  = t2[:,ki,kb].copy()
                    tau2_OovV  = unpack_tril(t2,nkpts,range(nkpts),ki,kb,kconserv[range(nkpts),kb,ki])
                    tau2_OovV[kb] += 2*einsum('id,lb->libd',t1[ki],t1[kb])
                    _WOvoV -= 0.5*einsum('dc,ibd->bic',oOVv_f,tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1))

                    for iterkj,kj in enumerate(ranges1):
                        if ki <= kj:
                            ka = kconserv[ki,kb,kj]
                            #tmp = einsum('bic,jac->ijab',_WOvoV,t2[kk_slice,kj,ka].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            tmp = einsum('bic,jac->ijab',_WOvoV,
                                         unpack_tril(t2,nkpts,kk_range,kj,ka,kconserv[kk_range,ka,kj]).transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            if ki == kj:
                                t2new_tril[tril_index(ki,kj),ka] -= tmp
                                t2new_tril[tril_index(ki,kj),kb] -= tmp.transpose(1,0,3,2)
                                #t2new[ki,kj,ka] -= tmp
                                #t2new[ki,kj,kb] -= tmp.transpose(1,0,3,2)
                            else:
                                t2new_tril[tril_index(ki,kj),ka] -= tmp
                                #t2new[ki,kj,ka] -= tmp
        loader.slave_finished()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Wovov (bi)', *cput2)

    cput2 = logger.process_clock(), logger.perf_counter()
    loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,nkpts_blksize,))
    loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

    # Making last of the Wovov terms (part 2/2)

    good2go = True
    while (good2go):
        good2go, data = loader.slave_set()
        if good2go is False:
            break
        ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
        if min(ranges0) >= max(ranges1): #continue if ki >= kj
            loader.slave_finished()
            continue

        s0,s1,s2 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1,ranges2)]

        eris_ovovRev_Xaj = _cp(eris.ovovRev[s1,s2,:])
        eris_ooovRev_Xaj = _cp(eris.ooovRev[s1,s2,:])
        eris_ovvvRev_Xaj = _cp(eris.ovvvRev[s1,s2,:])

        eris_vovvL1_ija = _cp(eris.vovvL1[s1,s2,:])

        for iterki,ki in enumerate(ranges0):
            for iterkj,kj in enumerate(ranges1):
                if ki < kj:
                    for iterka,ka in enumerate(ranges2):
                        kb = kconserv[ki,ka,kj]
                        # t1 with ovvv terms  (part 2/2)
                        tmp2 = (eris_vovvL1_ija[iterkj,iterka,iterki].transpose(3,2,1,0).conj() -
                                einsum('kajc,kb->bajc',eris_ovovRev_Xaj[iterkj,iterka,kb],t1[kb])) #ovvv[kj,ki,kb,ka]  ovov[kb,ka,kj,ki]
                        tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
                        t2new_tril[tril_index(ki,kj),ka] += tmp
                        #t2new[ki,kj,ka] += tmp

        for kblock in BLKSIZE2_ranges:
            kk_block_size = kblock[1]-kblock[0]
            kk_slice = slice(kblock[0],kblock[1])
            kk_range = range(kblock[0],kblock[1])

            oOVv   = numpy.empty((nkpts,kk_block_size,nocc,nocc,nvir,nvir),dtype=t2.dtype)

            for iterkj,kj in enumerate(ranges1):
                for iterka,ka in enumerate(ranges2):
                    kc_list = kconserv[kk_slice,kj,ka]
                    # Wovov term (kk,ka,kj,kc)
                    _WOvoV = _cp(eris_ovovRev_Xaj[iterkj,iterka,kk_slice]).transpose(2,3,0,1,4).reshape(nvir,nocc,-1)                          #ovov[*,ka,kj,*]
                    _WOvoV -= einsum('ljc,la->ajc',eris_ooovRev_Xaj[iterkj,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nocc,nocc,-1),t1[ka]) #ooov[*,ka,kj,*]
                    _WOvoV += einsum('adc,jd->ajc',eris_ovvvRev_Xaj[iterkj,iterka,kk_slice].transpose(2,3,0,1,4).reshape(nvir,nvir,-1),t1[kj]) #ovvv[*,ka,kj,*]
                    #
                    # Filling in the oovv terms
                    #
                    for iterkk,kk in enumerate(kk_range):
                        oOVv[:,iterkk] = _cp(eris.oovv[:,kk,kc_list[iterkk]])
                    oOVv_f = oOVv.transpose(0,2,5,1,3,4).reshape(nocc*nvir*nkpts,nocc*nvir*kk_block_size)

                    #tau2_OovV  = t2[:,kj,ka].copy()
                    tau2_OovV  = unpack_tril(t2,nkpts,range(nkpts),kj,ka,kconserv[range(nkpts),ka,kj])
                    tau2_OovV[ka] += 2*einsum('jd,la->ljad',t1[kj],t1[ka])
                    _WOvoV -= 0.5*einsum('dc,jad->ajc',oOVv_f,tau2_OovV.transpose(2,3,0,1,4).reshape(nocc,nvir,-1))

                    for iterki,ki in enumerate(ranges0):
                        if ki < kj:
                            kb = kconserv[ki,ka,kj]
                            #tmp = einsum('ajc,ibc->ijab',_WOvoV,t2[kk_slice,ki,kb].transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            tmp = einsum('ajc,ibc->ijab', _WOvoV,
                                         unpack_tril(t2,nkpts,kk_range,ki,kb,kconserv[kk_range,kb,ki]).transpose(2,3,0,1,4).reshape(nocc,nvir,-1))
                            t2new_tril[tril_index(ki,kj),ka] -= tmp
                            #t2new[ki,kj,ka] -= tmp
        loader.slave_finished()
    comm.Barrier()
    cput2 = log.timer_debug1('transforming Wovov (aj)', *cput2)

    comm.Barrier()
    safeAllreduceInPlace(comm, t2new_tril)

    for ki in range(nkpts):
        ka = ki
        # Remove zero/padded elements from denominator
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        t1new[ki] /= eia

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        if ki > kj:
            continue

        kb = kconserv[ki, ka, kj]
        # For LARGE_DENOM, see t1new update above
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

        ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
        ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        t2new_tril[tril_index(ki,kj),ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    comm.Barrier()
    return t1new, t2new_tril


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    fock = eris.fock
    e = 0.0 + 1j*0.0
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e += einsum('ijab,ijab', 2*tau[ki,kj,ka], eris.oovv[ki,kj,ka])
                e += einsum('ijab,ijba',  -tau[ki,kj,ka], eris.oovv[ki,kj,kb])
    e /= nkpts
    return e.real

def energy_tril(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.kconserv
    fock = eris.fock
    e = numpy.array(0.0,dtype=numpy.complex128)
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            if ki <= kj:
                t1t1[tril_index(ki,kj),ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                if ki <= kj:
                    kb = kconserv[ki,ka,kj]
                    e += einsum('ijab,ijab', tau[tril_index(ki,kj),ka],
                                (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
                if kj < ki:
                    kb = kconserv[ki,ka,kj]
                    e += einsum('ijab,ijab', tau[tril_index(kj,ki),kb].transpose(1,0,3,2),
                                (2.*eris.oovv[ki,kj,ka]-eris.oovv[ki,kj,kb].transpose(0,1,3,2)))
    comm.Barrier()
    e /= nkpts
    return e.real

def _update_procs_mf(mf):
    '''Update mean-field objects to be the same on all processors'''
    mf1 = copy.copy(mf)

    mo_coeff  = comm.bcast(mf.mo_coeff, root=0)
    mo_energy = comm.bcast(mf.mo_energy, root=0)
    mo_occ    = comm.bcast(mf.mo_occ, root=0)
    kpts      = comm.bcast(mf.kpts, root=0)

    mf1.mo_coeff = mo_coeff
    mf1.mo_energy = mo_energy
    mf1.mo_occ = mo_occ
    mf1.kpts  = kpts
    comm.Barrier()
    return mf1

class RCCSD(pyscf.pbc.cc.kccsd_rhf.RCCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        mf = _update_procs_mf(mf)
        pyscf.pbc.cc.kccsd_rhf.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kconserv = kpts_helper.get_kconserv(mf.cell, mf.kpts)
        self.__imds__ = None

    def _init_amps_tril(self, eris):
        time0 = logger.process_clock(), logger.perf_counter()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        tril_shape = ((nkpts)*(nkpts+1))//2
        t2_tril = numpy.zeros((tril_shape,nkpts,nocc,nocc,nvir,nvir),dtype=numpy.complex128)
        local_mp2 = numpy.array(0.0,dtype=numpy.complex128)
        self.emp2 = 0
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        mo_e_o = [e[:nocc] for e in eris.mo_energy]
        mo_e_v = [e[nocc:] for e in eris.mo_energy]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        kconserv = self.kconserv
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(1,1,nkpts,))
        loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
            for ki in ranges0:
                for kj in ranges1:
                    if ki <= kj:
                        for ka in ranges2:
                            kb = kconserv[ki,ka,kj]
                            # For discussion of LARGE_DENOM, see t1new update above
                            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
                            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
                            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
                            eijab = eia[:, None, :, None] + ejb[:, None, :]

                            oovv_ijab = numpy.array(eris.oovv[ki,kj,ka])
                            oovv_ijba = numpy.array(eris.oovv[ki,kj,kb]).transpose(0,1,3,2)
                            woovv = 2.*oovv_ijab - oovv_ijba

                            t2_tril[tril_index(ki,kj),ka] = numpy.conj(oovv_ijab / eijab)
                            local_mp2 += numpy.dot(t2_tril[tril_index(ki,kj),ka].flatten(),woovv.flatten())
                    if kj < ki:
                        for ka in ranges2:
                            kb = kconserv[ki,ka,kj]
                            # For discussion of LARGE_DENOM, see t1new update above
                            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
                            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
                            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
                            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
                            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
                            eijab = eia[:, None, :, None] + ejb[:, None, :]

                            idx = abs(eijab) < LOOSE_ZERO_TOL
                            eijab[idx] = LARGE_DENOM

                            oovv_ijab = numpy.array(eris.oovv[ki,kj,ka])
                            oovv_ijba = numpy.array(eris.oovv[ki,kj,kb]).transpose(0,1,3,2)
                            woovv = 2.*oovv_ijab - oovv_ijba

                            tmp = numpy.conj(oovv_ijab / eijab)
                            local_mp2 += numpy.dot(tmp.flatten(),woovv.flatten())
            loader.slave_finished()

        comm.Allreduce(MPI.IN_PLACE, local_mp2, op=MPI.SUM)
        safeAllreduceInPlace(comm, t2_tril)
        self.emp2 = local_mp2.real
        self.emp2 /= nkpts

        if rank == 0:
            logger.info(self, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
            logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2_tril

    def _init_amps(self, eris):
        time0 = logger.process_clock(), logger.perf_counter()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        self.emp2 = 0
        eris_oovv = eris.oovv.copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        mo_e_o = [e[:nocc] for e in eris.mo_energy]
        mo_e_v = [e[nocc:] for e in eris.mo_energy]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        kconserv = self.kconserv
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki,ka,kj]
            # For discussion of LARGE_DENOM, see t1new update above
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            woovv[ki,kj,ka] = (2*eris_oovv[ki,kj,ka] - eris_oovv[ki,kj,kb].transpose(0,1,3,2))
            t2[ki,kj,ka] = eris_oovv[ki,kj,ka] / eijab

        t2 = numpy.conj(t2)
        emp2 = numpy.einsum('pqrijab,pqrijab',t2,woovv).real
        emp2 /= nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def init_amps(self, eris):
        return self._init_amps_tril(eris)

    get_normt_diff = get_normt_diff

    def energy(self, t1, t2, eris):
        return energy_tril(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)

    def ipccsd_diag(self):
        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ip(self)
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = -numpy.diag(imds.Loo[kshift])

        Hr2 = numpy.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=t1.dtype)
        mem = 0.5e9
        pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
        nkpts_blksize  = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize2,nkpts_blksize,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1)]

            for iterki,ki in enumerate(ranges0):
                for iterkj,kj in enumerate(ranges1):
                    kb = kconserv[ki,kshift,kj]

                    Woooo_iji = _cp(imds.Woooo[ki,kj,ki])
                    Wvoov_bjj = _cp(imds.Wvoov[kb,kj,kj])
                    Wovov_jbj = _cp(imds.Wovov[kj,kb,kj])
                    Wovov_ibi = _cp(imds.Wovov[ki,kb,ki])

                    for b in range(nvir):
                        Hr2[ki,kj,:,:,b] = imds.Lvv[kb,b,b]
                    for i in range(nocc):
                        Hr2[ki,kj,i,:,:] -= imds.Loo[ki,i,i]
                    for j in range(nocc):
                        Hr2[ki,kj,:,j,:] -= imds.Loo[kj,j,j]
                    for i in range(nocc):
                        for j in range(nocc):
                            Hr2[ki,kj,i,j,:] += Woooo_iji[i,j,i,j]
                    for j in range(nocc):
                        for b in range(nvir):
                            Hr2[ki,kj,:,j,b] += 2.*Wvoov_bjj[b,j,j,b]
                            Hr2[ki,kj,j,j,b] += -Wvoov_bjj[b,j,j,b]
                            Hr2[ki,kj,:,j,b] += -Wovov_jbj[j,b,j,b]

                            Hr2[ki,kj,j,:,b] += -Wovov_ibi[j,b,j,b]

                            for i in range(nocc):
                                kd = kconserv[kj,kshift,ki]
                                Hr2[ki,kj,i,j,b] += -numpy.dot(unpack_tril(t2,nkpts,ki,kj,kshift,kconserv[ki,kshift,kj])[i,j,:,b],
                                                               -2.*imds.Woovv[kj,ki,kd,j,i,b,:] + imds.Woovv[ki,kj,kd,i,j,b,:])

            loader.slave_finished()
        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd(self, nroots=2*4, kptlist=None):
        time0 = logger.process_clock(), logger.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ip()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ip(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        evals = np.zeros((len(kptlist),nroots))
        evecs = np.zeros((len(kptlist),nroots,size), np.complex128)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            diag = self.ipccsd_diag()
            diag = self.mask_frozen_ip(diag, kshift, const=LARGE_DENOM)
            # Initial guess from file
            amplitude_filename = "__ripccsd" + str(kshift) + "__.hdf5"
            rsuccess, x0 = read_eom_amplitudes((nroots,size),filename=amplitude_filename)
            if x0 is not None:
                x0 = x0.T
            #if not rsuccess:
            #    x0 = np.zeros_like(diag)
            #    x0[np.argmin(diag)] = 1.0

            conv, evals_k, evecs_k = eigs(self.ipccsd_matvec, size, nroots, x0=x0, Adiag=diag, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k.T

            write_eom_amplitudes(evecs[k],filename=amplitude_filename)
        time0 = log.timer_debug1('converge ip-ccsd', *time0)
        comm.Barrier()
        return evals.real, evecs

    restore_from_diis_ = restore_from_diis_

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        if rank == 0:
            if (adiis and
                istep >= self.diis_start_cycle and
                abs(de) < self.diis_start_energy_diff):
                vec = self.amplitudes_to_vector(t1, t2)
                t1, t2 = self.vector_to_amplitudes(adiis.update(vec))
                logger.debug1(self, 'DIIS for step %d', istep)
        safeBcastInPlace(MPI.COMM_WORLD, t1)
        safeBcastInPlace(MPI.COMM_WORLD, t2)
        return t1, t2

    def ipccsd_matvec(self, vector):
        kshift = self.kshift
        # Ref: Z. Tu, F. Wang, and X. Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        r1,r2 = self.vector_to_amplitudes_ip(vector)
        r1 = comm.bcast(r1, root=0)
        r2 = comm.bcast(r2, root=0)

        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ip(self)
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = numpy.zeros(r1.shape,dtype=t1.dtype)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,))
        loader.set_ranges((range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0 = loader.get_blocks_from_data(data)

            s0 = slice(min(ranges0),max(ranges0)+1)

            Wooov_Xls = _cp(imds.Wooov[:,s0,kshift])
            Wooov_lXs = _cp(imds.Wooov[s0,:,kshift])

            for iterkl,kl in enumerate(ranges0):
                Hr1 += einsum('ld,ild->i',imds.Fov[kl],2.*r2[kshift,kl]-r2[kl,kshift].transpose(1,0,2))
                Hr1 += einsum('xklid,xkld->i',-2.*Wooov_Xls[:,iterkl]+Wooov_lXs[iterkl,:].transpose(0,2,1,3,4),r2[:,kl])
            loader.slave_finished()
        comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 -= einsum('ki,k->i',imds.Loo[kshift],r1)

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,1,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1)]
            Wovoo_sXi  = _cp(imds.Wovoo[kshift,:,s0])
            WooooS_Xij = _cp(imds.WooooS[:,s0,s1])

            tmp = numpy.zeros(nvir,dtype=t2.dtype)
            for kl in range(nkpts):
                kk_list = range(nkpts)
                kd_list = kconserv[kl,kshift,kk_list]
                tmp += einsum('lc,l->c', (2.*imds.Woovv[kl,kk_list,kd_list].transpose(1,0,2,3,4) -
                                          imds.Woovv[kk_list,kl,kd_list].transpose(2,0,1,3,4)).reshape(-1,nvir),
                                         r2[kk_list,kl].transpose(2,0,1,3).reshape(-1))
            for iterki, ki in enumerate(ranges0):
                for iterkj, kj in enumerate(ranges1):
                    Hr2[ki,kj] += -einsum('c,ijcb->ijb',tmp,unpack_tril(t2,nkpts,ki,kj,kshift,kconserv[ki,kshift,kj]))

            for iterki, ki in enumerate(ranges0):
                for iterkj, kj in enumerate(ranges1):
                    kb = kconserv[ki,kshift,kj]
                    Hr2[ki,kj] += einsum('bd,ijd->ijb',imds.Lvv[kb],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('li,ljb->ijb',imds.Loo[ki],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('lj,ilb->ijb',imds.Loo[kj],r2[ki,kj])
                    Hr2[ki,kj] -= einsum('kbij,k->ijb',Wovoo_sXi[kb,iterki],r1)

                    kl_list = range(nkpts)
                    kk_list = kconserv[ki,kl_list,kj]
                    Hr2[ki,kj] += einsum('klij,klb->ijb',
                                         WooooS_Xij[kl_list,iterki,iterkj].transpose(1,0,2,3,4).reshape(nocc,nocc*nkpts,nocc,nocc),
                                         r2[kk_list,kl_list].transpose(1,0,2,3).reshape(nocc,nocc*nkpts,nvir))

            Wovov_Xbi = _cp(imds.Wovov[:,s1,s0])

            for iterki,ki in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[kshift,ki,kb]
                    Hr2[ki,kj] += -einsum('lbid,ljd->ijb',Wovov_Xbi[:,iterkb,iterki].reshape(nocc*nkpts,nvir,nocc,nvir),
                                                          r2[:,kj].reshape(nocc*nkpts,nocc,nvir))
            Wvoov_bXj = _cp(imds.Wvoov[s1,:,s0])
            Wovov_Xbj = _cp(imds.Wovov[:,s1,s0])

            for iterkj,kj in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    ki = kconserv[kshift,kj,kb]
                    Hr2[ki,kj] += einsum('bljd,ild->ijb',
                                         Wvoov_bXj[iterkb,:,iterkj].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                         (2.*r2[ki,:].transpose(1,0,2,3)-
                                          r2[:,ki].transpose(2,0,1,3)).reshape(nocc,nocc*nkpts,nvir))
                    Hr2[ki,kj] += -einsum('lbjd,ild->ijb',
                                          Wovov_Xbj[:,iterkb,iterkj].reshape(nocc*nkpts,nvir,nocc,nvir),
                                          r2[ki,:].transpose(1,0,2,3).reshape(nocc,nocc*nkpts,nvir))  #typo in nooijen's paper
            loader.slave_finished()
        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        return vector

    def lipccsd(self, nroots=2*4, kptlist=None):
        time0 = logger.process_clock(), logger.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ip()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ip(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        evals = np.zeros((len(kptlist),nroots))
        evecs = np.zeros((len(kptlist),nroots,size), np.complex128)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            diag = self.ipccsd_diag()
            # Initial guess from file
            amplitude_filename = "__lipccsd" + str(kshift) + "__.hdf5"
            rsuccess, x0 = read_eom_amplitudes((nroots,size),filename=amplitude_filename)
            if x0 is not None:
                x0 = x0.T
            #if not rsuccess:
            #    x0 = np.zeros_like(diag)
            #    x0[np.argmin(diag)] = 1.0

            conv, evals_k, evecs_k = eigs(self.lipccsd_matvec, size, nroots, x0=x0, Adiag=diag, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k.T

            write_eom_amplitudes(evecs[k],filename=amplitude_filename)
        time0 = log.timer_debug1('converge ip-ccsd', *time0)
        comm.Barrier()
        return evals.real, evecs

    def lipccsd_matvec(self, vector):
        # Ref: Z. Tu, F. Wang, and X. Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        kshift = self.kshift
        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        r1,r2 = self.vector_to_amplitudes_ip(vector)
        r1 = comm.bcast(r1, root=0)
        r2 = comm.bcast(r2, root=0)

        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kconserv = self.kconserv

        if not self.made_ip_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ip(self)
            self.made_ip_imds = True

        imds = self.imds

        Hr1 = numpy.zeros(r1.shape,dtype=t1.dtype)
        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)

        def mem_usage_ovoo(nocc, nvir, nkpts):
            return nocc**3 * nvir**1 * 16
        array_size = [nkpts,nkpts]
        #chunk_size = get_max_blocksize_from_mem(0.3e9, mem_usage_ovoo(nocc,nvir,nkpts),
        #                                        array_size, priority_list=[1,1])
        #task_list = generate_task_list(chunk_size,array_size)
        task_list = generate_max_task_list(array_size,blk_mem_size=mem_usage_ovoo(nocc,nvir,nkpts),priority_list=[1,1])

        for kbrange, kirange in mpi.work_stealing_partition(task_list):
            Wovoo_sbi = _cp(imds.Wovoo[kshift,slice(*kbrange),slice(*kirange)])

            for iterkb, kb in enumerate(range(*kbrange)):
                for iterki, ki in enumerate(range(*kirange)):
                    kj = kconserv[kshift,ki,kb]
                    Hr1 -= einsum('kbij,ijb->k',Wovoo_sbi[iterkb,iterki],r2[ki,kj])

        comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 -= einsum('ki,i->k',imds.Loo[kshift],r1)
        #Hr1 = ( - einsum('ki,i->k',Loo,r1)
        #        - einsum('kbij,ijb->k',Wovoo,r2)
        #        )

        # Using same task_list as before
        for klrange, kkrange in mpi.work_stealing_partition(task_list):
            Wooov_kls = _cp(imds.Wooov[slice(*kkrange),slice(*klrange),kshift])
            Wooov_lks = _cp(imds.Wooov[slice(*klrange),slice(*kkrange),kshift])

            for iterkk, kk in enumerate(range(*kkrange)):
                for iterkl, kl in enumerate(range(*klrange)):
                    kd = kconserv[kk,kshift,kl]
                    Hr2[kk,kl] -= 2.*einsum('klid,i->kld',Wooov_kls[iterkk,iterkl],r1)
                    Hr2[kk,kl] += einsum('lkid,i->kld',Wooov_lks[iterkl,iterkk],r1)
                    Hr2[kk,kl] -= einsum('ki,ild->kld',imds.Loo[kk],r2[kk,kl])
                    Hr2[kk,kl] -= einsum('lj,kjd->kld',imds.Loo[kl],r2[kk,kl])
                    Hr2[kk,kl] += einsum('bd,klb->kld',imds.Lvv[kd],r2[kk,kl])
                    Hr2[kk,kshift] -= (kk==kd)*einsum('kd,l->kld',imds.Fov[kk],r1)
                    Hr2[kshift,kl] += (kl==kd)*2.*einsum('ld,k->kld',imds.Fov[kl],r1)

        def mem_usage_ovvok(nocc, nvir, nkpts):
            return nocc**2 * nvir**2 * nkpts *  16
        array_size = [nkpts,nkpts]
        #chunk_size = get_max_blocksize_from_mem(0.3e9, 2.*mem_usage_ovvok(nocc,nvir,nkpts),
        #                                        array_size, priority_list=[1,1])
        #task_list = generate_task_list(chunk_size,array_size)
        task_list = generate_max_task_list(array_size,blk_mem_size=mem_usage_ovvok(nocc,nvir,nkpts),priority_list=[1,1])

        for kbrange, klrange in mpi.work_stealing_partition(task_list):

            Wvoov_blX = _cp(imds.Wvoov[slice(*kbrange),slice(*klrange),:])
            Wovov_lbX = _cp(imds.Wovov[slice(*klrange),slice(*kbrange),:])

            for iterkb, kb in enumerate(range(*kbrange)):
                for iterkl, kl in enumerate(range(*klrange)):
                    for iterkj, kj in enumerate(range(nkpts)):
                        kd = kconserv[kb,kj,kl]
                        kk = kconserv[kshift,kl,kd]
                        tmp = einsum('bljd,kjb->kld',Wvoov_blX[iterkb,iterkl,kj],r2[kk,kj])
                        Hr2[kk,kl] += 2.*tmp
                        Hr2[kk,kl] -= einsum('lbjd,kjb->kld', Wovov_lbX[iterkl,iterkb,kj],
                                             r2[kk,kj])  # typo in nooijen's paper

                        # Notice we switch around the variable kk and kl
                        kd = kconserv[kl,kj,kb]
                        kk = kconserv[kshift,kl,kd]
                        Hr2[kl,kk] -= einsum('kbjd,jlb->kld',Wovov_lbX[iterkl,iterkb,kj],r2[kj,kk])
                        Hr2[kl,kk] -= einsum('bkjd,ljb->kld',Wvoov_blX[iterkb,iterkl,kj],r2[kk,kj])

        def mem_usage_ovvok(nocc, nvir, nkpts):
            return nocc**2 * nvir**2 * nkpts *  16
        array_size = [nkpts,nkpts]
        #chunk_size = get_max_blocksize_from_mem(0.3e9, 3.*mem_usage_ovvok(nocc,nvir,nkpts),
        #                                        array_size, priority_list=[1,1])
        #task_list = generate_task_list(chunk_size,array_size)
        task_list = generate_max_task_list(array_size,blk_mem_size=mem_usage_ovvok(nocc,nvir,nkpts),priority_list=[1,1])

        # TODO tmp2 only needs to create tmp2[kshift], but should wait for the fix in the mpi.stealing
        # as defined for the analogous quantity in the ipccsd_matvec
        tmp2 = numpy.zeros((nkpts,nvir),dtype=t1.dtype)
        for kirange, kjrange in mpi.work_stealing_partition(task_list):
            for iterki, ki in enumerate(range(*kirange)):
                for iterkj, kj in enumerate(range(*kjrange)):
                    for iterkc, kc in enumerate(range(nkpts)):
                        t2_tmp = unpack_tril(t2,nkpts,ki,kj,kc,kconserv[ki,kc,kj])
                        tmp2[kc] += einsum('ijcb,ijb->c',t2_tmp,r2[ki,kj])
        comm.Allreduce(MPI.IN_PLACE, tmp2, op=MPI.SUM)

        for kkrange, klrange in mpi.work_stealing_partition(task_list):

            Woooo_klX = _cp(imds.Woooo[slice(*kkrange),slice(*klrange),:])
            Woovv_klX = _cp(imds.Woovv[slice(*kkrange),slice(*klrange),:])

            for iterkk, kk in enumerate(range(*kkrange)):
                for iterkl, kl in enumerate(range(*klrange)):
                    for iterki, ki in enumerate(range(nkpts)):
                        kj = kconserv[kk,ki,kl]
                        Hr2[kk,kl] += einsum('klij,ijd->kld',Woooo_klX[iterkk,iterkl,ki],r2[ki,kj])
                    kd = kconserv[kk,kshift,kl]
                    tmp3 = einsum('kldc,c->kld',Woovv_klX[iterkk,iterkl,kd],tmp2[kshift])
                    Hr2[kk,kl] +=    tmp3
                    Hr2[kl,kk] -= 2.*tmp3.transpose(1,0,2) # Notice change of kl,kk in Hr2

        #tmp = einsum('ijcb,ijb->c',t2,r2)
        #Hr2 = ( - einsum('kd,l->kld',Fov,r1)
        #        + 2.*einsum('ld,k->kld',Fov,r1)
        #        - 2.*einsum('klid,i->kld',Wooov,r1)
        #        + einsum('lkid,i->kld',Wooov,r1)
        #        - einsum('ki,ild->kld',Loo,r2)
        #        - einsum('lj,kjd->kld',Loo,r2)
        #        + einsum('bd,klb->kld',Lvv,r2)
        #        + 2.*einsum('lbdj,kjb->kld',Wovvo,r2)
        #        - einsum('kbdj,ljb->kld',Wovvo,r2)
        #        - einsum('lbjd,kjb->kld',Wovov,r2) #typo in nooijen's paper
        #        + einsum('klij,ijd->kld',Woooo,r2)
        #        - einsum('kbid,ilb->kld',Wovov,r2)
        #        + einsum('kldc,c->kld',Woovv,tmp)
        #        - 2.*einsum('lkdc,c->kld',Woovv,tmp)
        #        )
        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        vector = self.mask_frozen_ip(vector, kshift, const=0.0)
        return vector

    def vector_size_ip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nocc + nkpts ** 2 * nocc ** 2 * nvir
        return size

    mask_frozen_ip = mask_frozen_ip

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
        #r2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), vector.dtype)
        #index = nocc
        #for ki in range(nkpts):
        #    for kj in range(nkpts):
        #        for i in range(nocc):
        #            for j in range(nocc):
        #                for a in range(nvir):
        #                    r2[ki,kj,i,j,a] =  vector[index]
        #                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size = nocc + nkpts*nkpts*nocc*nocc*nvir

        vector = np.zeros((size), r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nkpts*nkpts*nocc*nocc*nvir)
        #index = nocc
        #for ki in range(nkpts):
        #    for kj in range(nkpts):
        #        for i in range(nocc):
        #            for j in range(nocc):
        #                for a in range(nvir):
        #                    vector[index] = r2[ki,kj,i,j,a]
        #                    index += 1
        return vector

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.kconserv
        eris = self.eris
        kshift = self.kshift
        t2 = self.t2
        foo = eris.fock[:,:nocc,:nocc]
        fvv = eris.fock[:,nocc:,nocc:]

        e = []
        assert len(ipccsd_evecs.shape) == 2  # Done at a single k-point, kshift
        for _eval, _evec, _levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ip(_levec)
            r1,r2 = self.vector_to_amplitudes_ip(_evec)

            # Transposing the l2 operator

            l2_t = numpy.zeros_like(l2)
            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = kconserv[ki,kshift,kj]
                    l2_t[ki,kj] = l2[kj,ki].transpose(1,0,2)

            # Normalization constant for left-hand eigenvector
            ldotr = numpy.dot(l1.ravel(),r1.ravel()) + numpy.dot(l2.ravel(),r2.ravel())
            l2 = (l2 + 2.*l2_t)/3.
            l2_t = None

            l1 /= ldotr
            l2 /= ldotr
            l1 = l1.reshape(-1)
            r1 = r1.reshape(-1)

            lijkab_tmp = numpy.zeros((nocc,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            rijkab_tmp = numpy.zeros((nocc,nocc,nocc,nvir,nvir),dtype=t2.dtype)
            def mem_usage_ovvvkk(nocc, nvir, nkpts):
                return nocc * nvir**3 * nkpts * 16
            array_size = [nkpts,nkpts]
            # TODO: figure out a good chunk size
            #task_list = generate_max_task_list(array_size,priority_list=[1,1])
            task_list = generate_max_task_list(array_size,blk_mem_size=1e9*2.*mem_usage_ovvvkk(nocc,nvir,nkpts),priority_list=[1,1])

            star_energy = numpy.array(0.0 + 1j*0.0)
            for kirange, kjrange in mpi.work_stealing_partition(task_list):
                oovv_ijX = _cp(eris.oovv[slice(*kirange),slice(*kjrange),range(nkpts)])
                oovv_jiX = _cp(eris.oovv[slice(*kjrange),slice(*kirange),range(nkpts)])

                kklist = kpts_helper.get_kconserv3(self._scf.cell, self.kpts, [range(nkpts),range(nkpts),kshift,range(*kirange),range(*kjrange)])

                for iterki, ki in enumerate(range(*kirange)):
                    for iterkj, kj in enumerate(range(*kjrange)):
                        for iterka, ka in enumerate(range(nkpts)):

                            # Starting the left amplitude equations

                            for iterkb, kb in enumerate(range(nkpts)):
                                if ka > kb:
                                    continue
                                fac = (2.0 - float(ka==kb))

                                if ka == kb:
                                    if ki > kj:
                                        continue
                                    else:
                                        fac *= (2.0 - float(ki==kj))

                                lijkab_tmp *= 0.0
                                rijkab_tmp *= 0.0
                                kk = kklist[ka,kb,iterki,iterkj]

                                # (i,j,k) -> (i,j,k)
                                if kk == kshift and kb == kconserv[ki,ka,kj]:
                                    tmp = 0.5*einsum('ijab,k->ijkab',oovv_ijX[iterki,iterkj,iterka],l1)
                                    lijkab_tmp += 8.* tmp

                                if kk == kshift and kb == kconserv[ki,ka,kj]:
                                    # (j,i,k) -> (i,j,k)
                                    tmp = 0.5*einsum('jiab,k->ijkab',oovv_jiX[iterkj,iterki,ka],l1)
                                    lijkab_tmp -= 4.* tmp

                                # (k,j,i) -> (i,j,k)
                                if ki == kshift and kb == kconserv[kj,ka,kk]:
                                    tmp = 0.5*einsum('kjab,i->ijkab',eris.oovv[kk,kj,ka],l1)
                                    lijkab_tmp -= 4.* tmp

                                # (i,k,j) -> (i,j,k)
                                if kj == kshift and kb == kconserv[ki,ka,kk]:
                                    tmp = 0.5*einsum('ikab,j->ijkab',eris.oovv[ki,kk,ka],l1)
                                    lijkab_tmp -= 4.* tmp

                                # (j,k,i) -> (i,j,k)
                                if kj == kshift and kb == kconserv[ki,ka,kk]:
                                    tmp = 0.5*einsum('kiab,j->ijkab',eris.oovv[kk,ki,ka],l1)
                                    lijkab_tmp += 2.* tmp

                                # (k,i,j) -> (i,j,k)
                                if ki == kshift and kb == kconserv[kj,ka,kk]:
                                    tmp = 0.5*einsum('jkab,i->ijkab',eris.oovv[kj,kk,ka],l1)
                                    lijkab_tmp += 2.* tmp

                                # Beginning of ovvv terms

                                ke = kconserv[ka,ki,kb]
                                tmp = einsum('ieab,jke->ijkab',eris.ovvv[ki,ke,ka],2.*l2[kj,kk]-l2[kk,kj].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                ke = kconserv[ka,kj,kb]
                                tmp = einsum('jeba,ike->ijkab',eris.ovvv[kj,ke,kb],2.*l2[ki,kk]-l2[kk,ki].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                # (j,i,k) -> (i,j,k)
                                ke = kconserv[ka,kj,kb]
                                tmp = einsum('jeab,ike->ijkab',eris.ovvv[kj,ke,ka],2.*l2[ki,kk]-l2[kk,ki].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                ke = kconserv[ka,ki,kb]
                                tmp = einsum('ieba,jke->ijkab',eris.ovvv[ki,ke,kb],2.*l2[kj,kk]-l2[kk,kj].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                # (k,j,i) -> (i,j,k)
                                ke = kconserv[ki,kshift,kj]
                                tmp = einsum('keab,jie->ijkab',eris.ovvv[kk,ke,ka],2.*l2[kj,ki]-l2[ki,kj].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                ke = kconserv[ki,kshift,kj]
                                tmp = einsum('keba,ije->ijkab',eris.ovvv[kk,ke,kb],2.*l2[ki,kj]-l2[kj,ki].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                # ooov part 1

                                km = kconserv[kshift,ki,ka]
                                tmp = -einsum('kjmb,ima->ijkab',eris.ooov[kk,kj,km],2.*l2[ki,km]-l2[km,ki].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                km = kconserv[kshift,kj,kb]
                                tmp = -einsum('kima,jmb->ijkab',eris.ooov[kk,ki,km],2.*l2[kj,km]-l2[km,kj].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                # (j,i,k) -> (i,j,k)
                                km = kconserv[kshift,kj,ka]
                                tmp = -einsum('kimb,jma->ijkab',eris.ooov[kk,ki,km],2.*l2[kj,km]-l2[km,kj].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                km = kconserv[kshift,ki,kb]
                                tmp = -einsum('kjma,imb->ijkab',eris.ooov[kk,kj,km],2.*l2[ki,km]-l2[km,ki].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                # (k,j,i) -> (i,j,k)
                                km = kconserv[kshift,kj,kb]
                                tmp = -einsum('ikma,jmb->ijkab',eris.ooov[ki,kk,km],2.*l2[kj,km]-l2[km,kj].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                km = kconserv[kshift,ki,ka]
                                tmp = -einsum('jkmb,ima->ijkab',eris.ooov[kj,kk,km],2.*l2[ki,km]-l2[km,ki].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                # ooov part 2

                                km = kconserv[ki,kb,kj]
                                tmp = -einsum('ijmb,mka->ijkab',eris.ooov[ki,kj,km],2.*l2[km,kk]-l2[kk,km].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                km = kconserv[kj,ka,ki]
                                tmp = -einsum('jima,mkb->ijkab',eris.ooov[kj,ki,km],2.*l2[km,kk]-l2[kk,km].transpose(1,0,2))
                                lijkab_tmp += 2.*tmp

                                # (j,i,k) -> (i,j,k)
                                km = kconserv[ki,kb,kj]
                                tmp = -einsum('jimb,mka->ijkab',eris.ooov[kj,ki,km],2.*l2[km,kk]-l2[kk,km].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                km = kconserv[kj,ka,ki]
                                tmp = -einsum('ijma,mkb->ijkab',eris.ooov[ki,kj,km],2.*l2[km,kk]-l2[kk,km].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                # (k,j,i) -> (i,j,k)
                                km = kconserv[kshift,ki,kb]
                                tmp = -einsum('jkma,mib->ijkab',eris.ooov[kj,kk,km],2.*l2[km,ki]-l2[ki,km].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                                km = kconserv[kshift,kj,ka]
                                tmp = -einsum('ikmb,mja->ijkab',eris.ooov[ki,kk,km],2.*l2[km,kj]-l2[kj,km].transpose(1,0,2))
                                lijkab_tmp -= 1.*tmp

                            # Starting the right amplitude equations #

                                kk = kklist[ka,kb,iterki,iterkj]
                                km = kshift
                                ke = kconserv[ki,ka,kj]
                                tmp = einsum('mbke,m->bke',eris.ovov[km,kb,kk],r1)
                                tril = unpack_tril(t2,nkpts,ki,kj,ka,kconserv[ki,ka,kj])
                                tmp2 = -einsum('bke,ijae->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                ke = kconserv[ki,kb,kj]
                                tmp = einsum('make,m->ake',eris.ovov[km,ka,kk],r1)
                                tril = unpack_tril(t2,nkpts,kj,ki,kb,kconserv[kj,kb,ki])
                                tmp2 = -einsum('ake,jibe->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                kk = kklist[ka,kb,iterki,iterkj]
                                km = kshift
                                ke = kconserv[km,kj,kb]
                                tmp = einsum('mbej,m->bej',eris.ovvo[km,kb,ke],r1)
                                tril = unpack_tril(t2,nkpts,ki,kk,ka,kconserv[ki,ka,kk])
                                tmp2 = -einsum('bej,ikae->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                ke = kconserv[km,ki,ka]
                                tmp = einsum('maei,m->aei',eris.ovvo[km,ka,ke],r1)
                                tril = unpack_tril(t2,nkpts,kj,kk,kb,kconserv[kj,kb,kk])
                                tmp2 = -einsum('aei,jkbe->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                # ki is free, kj is free, ka is free, kk is free
                                # km, kb -> fixed
                                # (i,j,k) -> (i,j,k)
                                kk = kklist[ka,kb,iterki,iterkj]
                                km = kconserv[ka,ki,kb]
                                tmp = einsum('mnjk,n->mjk',eris.oooo[km,kshift,kj],r1)
                                tril = unpack_tril(t2,nkpts,ki,km,ka,kconserv[ki,ka,km])
                                tmp2 = einsum('mjk,imab->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                km = kconserv[kb,kj,ka]
                                tmp = einsum('mnik,n->mik',eris.oooo[km,kshift,ki],r1)
                                tril = unpack_tril(t2,nkpts,kj,km,kb,kconserv[kj,kb,km])
                                tmp2 = einsum('mik,jmba->ijkab',tmp,tril)
                                rijkab_tmp += tmp2

                                ke = kconserv[ka,ki,kb]
                                #if kk == kconserv[kshift,kj,ke]:
                                # Here we need vvov terms, but instead use ovvv.transpose(2,3,0,1).conj()
                                # terms, where the transpose was done in the einsum.
                                #
                                # See corresponding rccsd_eom equation
                                tmp2 = einsum('ieab,kje->ijkab',eris.ovvv[ki,ke,ka].conj(),r2[kk,kj])
                                rijkab_tmp += tmp2

                                ke = kconserv[kb,kj,ka]
                                tmp2 = einsum('jeba,kie->ijkab',eris.ovvv[kj,ke,kb].conj(),r2[kk,ki])
                                rijkab_tmp += tmp2

                                km = kconserv[kshift,ki,ka]
                                tmp2 = -einsum('kjmb,mia->ijkab',eris.ooov[kk,kj,km].conj(),r2[km,ki])
                                rijkab_tmp += tmp2

                                km = kconserv[kshift,kj,kb]
                                tmp2 = -einsum('kima,mjb->ijkab',eris.ooov[kk,ki,km].conj(),r2[km,kj])
                                rijkab_tmp += tmp2

                                km = kconserv[ki,kb,kj]
                                tmp2 = -einsum('mbij,kma->ijkab',eris.ovoo[km,kb,ki],r2[kk,km])
                                rijkab_tmp += tmp2

                                km = kconserv[kj,ka,ki]
                                tmp2 = -einsum('maji,kmb->ijkab',eris.ovoo[km,ka,kj],r2[kk,km])
                                rijkab_tmp += tmp2

                                eia = numpy.diagonal(foo[ki]).reshape(-1,1) - numpy.diagonal(fvv[ka])
                                eia += _eval

                                ejb = numpy.diagonal(foo[kj]).reshape(-1,1) - numpy.diagonal(fvv[kb])
                                eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)

                                eijkab = pyscf.lib.direct_sum('ijab,k->ijkab',eijab,numpy.diagonal(foo[kk]))
                                eijkab = 1./eijkab

                                star_energy += fac*0.5*einsum('ijkab,ijkab,ijkab',lijkab_tmp,rijkab_tmp,eijkab)

            comm.Allreduce(MPI.IN_PLACE, star_energy, op=MPI.SUM)
            if rank == 0:
                logger.info(self, 'EOM-IPCCSD* delta energy = %.15g (Imag = %.15g)',
                            star_energy.real,star_energy.imag)
            e.append(star_energy+_eval)

        return np.array(e)

    def eaccsd_diag(self):
        t1,t2 = self.t1, self.t2
        nkpts, nocc, nvir = t1.shape
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ea(self)
            self.made_ea_imds = True

        imds = self.imds

        Hr1 = numpy.diag(imds.Lvv[kshift])

        Hr2 = numpy.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=t1.dtype)
        mem = 0.5e9
        pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
        nkpts_blksize  = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize2,nkpts_blksize,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1)]

            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]

                    Wvvvv_aba    = _cp(imds.Wvvvv[ka,kb,ka])
                    WvoovR1_jbj  = _cp(imds.WvoovR1[kj,kb,kj])
                    WovovRev_jbj = _cp(imds.WovovRev[kj,kb,kj])
                    WovovRev_jaj = _cp(imds.WovovRev[kj,ka,kj])

                    for j in range(nocc):
                        Hr2[kj,ka,j,:,:] -= imds.Loo[kj,j,j]
                    for a in range(nvir):
                        Hr2[kj,ka,:,a,:] += imds.Lvv[ka,a,a]
                    for b in range(nvir):
                        Hr2[kj,ka,:,:,b] += imds.Lvv[kb,b,b]

                    for a in range(nvir):
                        for b in range(nvir):
                            Hr2[kj,ka,j,a,b] += Wvvvv_aba[a,b,a,b]

                    for j in range(nocc):
                        for b in range(nvir):
                            Hr2[kj,ka,j,:,b] += 2.*WvoovR1_jbj[b,j,j,b]
                            Hr2[kj,ka,j,:,b] += -WovovRev_jbj.transpose(1,0,3,2)[b,j,b,j]
                            Hr2[kj,ka,j,b,b] += -WvoovR1_jbj[b,j,j,b]

                            Hr2[kj,ka,j,b,:] += -WovovRev_jaj.transpose(1,0,3,2)[b,j,b,j]
                            for a in range(nvir):
                                Hr2[kj,ka,j,a,b] += numpy.dot(unpack_tril(t2,nkpts,kshift,kj,ka,kconserv[kshift,ka,kj])[:,j,a,b],
                                                              -2.*imds.Woovv[kshift,kj,ka,:,j,a,b] + imds.Woovv[kshift,kj,kb,:,j,b,a])

            loader.slave_finished()
        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd(self, nroots=2*4, kptlist=None):
        time0 = logger.process_clock(), logger.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ea()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ea(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        evals = np.zeros((len(kptlist),nroots))
        evecs = np.zeros((len(kptlist),nroots,size), np.complex128)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            diag = self.eaccsd_diag()
            diag = self.mask_frozen_ea(diag, kshift, const=LARGE_DENOM)
            # Initial guess from file
            amplitude_filename = "__reaccsd" + str(kshift) + "__.hdf5"
            rsuccess, x0 = read_eom_amplitudes((nroots,size),filename=amplitude_filename)
            if x0 is not None:
                x0 = x0.T
            #if not rsuccess:
            #    x0 = np.zeros_like(diag)
            #    x0[np.argmin(diag)] = 1.0

            conv, evals_k, evecs_k = eigs(self.eaccsd_matvec, size, nroots, x0=x0, Adiag=diag, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k.T

            write_eom_amplitudes(evecs[k],filename=amplitude_filename)
        time0 = log.timer_debug1('converge ea-ccsd', *time0)
        comm.Barrier()
        return evals.real, evecs

    def eaccsd_matvec(self, vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        kshift = self.kshift
        vector = self.mask_frozen_ea(vector, kshift, const=0.0)
        r1,r2 = self.vector_to_amplitudes_ea(vector)
        r1 = comm.bcast(r1, root=0)
        r2 = comm.bcast(r2, root=0)

        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ea(self)
            self.made_ea_imds = True

        imds = self.imds

        Hr1 = numpy.zeros(r1.shape,dtype=t1.dtype)
        mem = 0.5e9
        pre = 1.*nocc*nvir*nvir*nvir*nkpts*16
        nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize,))
        loader.set_ranges((range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0 = loader.get_blocks_from_data(data)

            s0 = slice(min(ranges0),max(ranges0)+1)

            Wvovv_slX = _cp(imds.Wvovv[kshift,s0,:])

            for iterkl,kl in enumerate(ranges0):
                Hr1 += 2.*einsum('ld,lad->a',imds.Fov[kl],r2[kl,kshift])
                Hr1 +=   -einsum('ld,lda->a',imds.Fov[kl],r2[kl,kl])
                kd_list = numpy.array(kconserv[kshift,range(nkpts),kl])
                Hr1 += einsum('alxcd,lxcd->a',
                              (2.*Wvovv_slX[iterkl,:].transpose(1,2,0,3,4) -
                               Wvovv_slX[iterkl,kd_list].transpose(1,2,0,4,3)),
                              r2[kl,:].transpose(1,0,2,3))
            loader.slave_finished()
        comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 += einsum('ac,c->a',imds.Lvv[kshift],r1)

        Hr2 = numpy.zeros(r2.shape,dtype=t1.dtype)
        mem = 0.5e9
        pre = 1.*nvir*nvir*nvir*nvir*nkpts*16
        nkpts_blksize  = min(max(int(numpy.floor(mem/pre)),1),nkpts)
        nkpts_blksize2 = min(max(int(numpy.floor(mem/(nkpts_blksize*pre))),1),nkpts)
        loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts_blksize2,nkpts_blksize,))
        loader.set_ranges((range(nkpts),range(nkpts),))

        good2go = True
        while (good2go):
            good2go, data = loader.slave_set()
            if good2go is False:
                break
            ranges0, ranges1 = loader.get_blocks_from_data(data)

            s0,s1 = [slice(min(x),max(x)+1) for x in (ranges0,ranges1)]

            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] -= einsum('lj,lab->jab',imds.Loo[kj],r2[kj,ka])
                    Hr2[kj,ka] += einsum('ac,jcb->jab',imds.Lvv[ka],r2[kj,ka])
                    Hr2[kj,ka] += einsum('bd,jad->jab',imds.Lvv[kb],r2[kj,ka])

            WvvvoR1_abX = _cp(imds.WvvvoR1[kshift,s0,s1])
            for iterka,ka in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[ka,kshift,kb]
                    Hr2[kj,ka] += einsum('abcj,c->jab',WvvvoR1_abX[iterka,iterkb],r1)

            WovovRev_Xaj = _cp(imds.WovovRev[s0,s1,:])
            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    kl_range = range(nkpts)
                    kd_range = kconserv[ka,kj,kl_range]
                    Hr2[kj,ka] += -einsum('axj,xb->jab',
                                          WovovRev_Xaj[iterkj,iterka,:].transpose(2,0,1,4,3).reshape(nvir,nocc*nvir*nkpts,nocc),
                                          r2[kl_range,kd_range].reshape(nkpts*nocc*nvir,nvir))

            tmp = numpy.zeros(nocc,dtype=t2.dtype)
            for kl in range(nkpts):
                kd_range = _cp(range(nkpts))
                kc_range = _cp(kconserv[kshift,kd_range,kl])
                tmp += einsum('kl,l->k',(2.*imds.Woovv[kshift,kl,kc_range].transpose(1,2,0,3,4)-
                                            imds.Woovv[kshift,kl,kd_range].transpose(1,2,0,4,3)).reshape(nocc,-1),
                                        r2[kl,kc_range].transpose(1,0,2,3).reshape(-1))

            for iterkj,kj in enumerate(ranges0):
                for iterka,ka in enumerate(ranges1):
                    kb = kconserv[kshift,ka,kj]
                    Hr2[kj,ka] += -einsum('k,kjab->jab', tmp, unpack_tril(t2,nkpts,kshift,kj,ka,kb))

            WovovRev_Xbj = _cp(imds.WovovRev[s0,s1,:])
            WvoovR1_bXj = _cp(imds.WvoovR1[s0,s1,:])
            for iterkj,kj in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    ka = kconserv[kshift,kb,kj]
                    Hr2[kj,ka] += -einsum('bldj,lad->jab',
                                          WovovRev_Xbj[iterkj,iterkb,:].transpose(2,0,1,4,3).reshape(nvir,nkpts*nocc,nvir,nocc),
                                          r2[:,ka].reshape(nkpts*nocc,nvir,nvir))
                    kl_range = _cp(range(nkpts))
                    kd_range = _cp(kconserv[kb,kj,kl_range])
                    Hr2[kj,ka] += einsum('bljd,lad->jab',
                                         WvoovR1_bXj[iterkj,iterkb,:].transpose(1,0,2,3,4).reshape(nvir,nocc*nkpts,nocc,nvir),
                                         (2.*r2[:,ka]-r2[kl_range,kd_range].transpose(0,1,3,2)).reshape(nocc*nkpts,nvir,nvir))

            Wvvvv_abX = _cp(imds.Wvvvv[s0,s1])
            for iterka,ka in enumerate(ranges0):
                for iterkb,kb in enumerate(ranges1):
                    kj = kconserv[ka,kshift,kb]
                    Hr2[kj,ka] += einsum('abx,jx->jab',
                                         Wvvvv_abX[iterka,iterkb,:].transpose(1,2,0,3,4).reshape(nvir,nvir,nvir*nkpts*nvir),
                                         r2[kj,:].transpose(1,0,2,3).reshape(nocc,nvir*nkpts*nvir))
            loader.slave_finished()

        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        vector = self.mask_frozen_ea(vector, kshift, const=0.0)
        return vector

    def leaccsd(self, nroots=2*4, kptlist=None):
        time0 = logger.process_clock(), logger.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if kptlist is None:
            kptlist = range(nkpts)
        size = self.vector_size_ea()
        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            nfrozen = np.sum(self.mask_frozen_ea(np.zeros(size, dtype=int), kshift, const=1))
            nroots = min(nroots, size - nfrozen)
        evals = np.zeros((len(kptlist),nroots))
        evecs = np.zeros((len(kptlist),nroots,size), np.complex128)

        for k,kshift in enumerate(kptlist):
            self.kshift = kshift
            diag = self.eaccsd_diag()
            # Initial guess from file
            amplitude_filename = "__leaccsd" + str(kshift) + "__.hdf5"
            rsuccess, x0 = read_eom_amplitudes((nroots,size),filename=amplitude_filename)
            if x0 is not None:
                x0 = x0.T
            #if not rsuccess:
            #    x0 = np.zeros_like(diag)
            #    x0[np.argmin(diag)] = 1.0

            conv, evals_k, evecs_k = eigs(self.leaccsd_matvec, size, nroots, x0=x0, Adiag=diag, verbose=self.verbose)

            evals_k = evals_k.real
            evals[k] = evals_k
            evecs[k] = evecs_k.T

            write_eom_amplitudes(evecs[k],amplitude_filename)
        time0 = log.timer_debug1('converge lea-ccsd', *time0)
        comm.Barrier()
        return evals.real, evecs

    def leaccsd_matvec(self, vector):
        # See relevant equations in cc/rccsd_slow.py
        # Does not follow the equations in Nooijen's paper
        # for eaccsd, uses a different left basis.
        r1,r2 = self.vector_to_amplitudes_ea(vector)
        r1 = comm.bcast(r1, root=0)
        r2 = comm.bcast(r2, root=0)

        t1,t2 = self.t1, self.t2
        nkpts,nocc,nvir = self.t1.shape
        nkpts = self.nkpts
        kshift = self.kshift
        kconserv = self.kconserv

        if not self.made_ea_imds:
            if not getattr(self, 'imds', None):
                self.imds = _IMDS(self)
            self.imds.make_ea(self)
            self.made_ea_imds = True

        imds = self.imds

        # Beginning HR1 multiplication
        Hr1 = numpy.zeros(r1.shape,dtype=t2.dtype)
        def mem_usage_vvvo(nocc, nvir, nkpts):
            return nocc**1 * nvir**3 * 16.
        array_size = [nkpts,nkpts]
        task_list = generate_max_task_list(array_size,blk_mem_size=mem_usage_vvvo(nocc,nvir,nkpts),priority_list=[1,1])
        for karange, kbrange in mpi.work_stealing_partition(task_list):
            WvvvoR1_sab = _cp(imds.WvvvoR1[kshift,slice(*karange),slice(*kbrange)])
            for iterka, ka in enumerate(range(*karange)):
                for iterkb, kb in enumerate(range(*kbrange)):
                    kj = kconserv[ka,kshift,kb]
                    Hr1 += einsum('abcj,jab->c',WvvvoR1_sab[iterka,iterkb],r2[kj,ka])
        comm.Allreduce(MPI.IN_PLACE, Hr1, op=MPI.SUM)
        Hr1 += einsum('ac,a->c',imds.Lvv[kshift],r1)

        # Beginning HR2 multiplication
        Hr2 = numpy.zeros(r2.shape,dtype=t2.dtype)
        # use same task list as before
        for klrange, kcrange in mpi.work_stealing_partition(task_list):
            Wvovv_slc = _cp(imds.Wvovv[kshift,slice(*klrange),slice(*kcrange)])
            for iterkl, kl in enumerate(range(*klrange)):
                for iterkc, kc in enumerate(range(*kcrange)):
                    kd = kconserv[kl,kc,kshift]
                    Hr2[kl,kc] +=  einsum('lad,ac->lcd',r2[kl,kc],imds.Lvv[kc])
                    Hr2[kl,kc] +=  einsum('lcb,bd->lcd',r2[kl,kc],imds.Lvv[kd])
                    Hr2[kl,kc] += -einsum('jcd,lj->lcd',r2[kl,kc],imds.Loo[kl])
                    tmp = einsum('a,alcd->lcd',r1,Wvovv_slc[iterkl,iterkc])
                    Hr2[kl,kc] += 2.*tmp
                    Hr2[kl,kd] +=   -tmp.transpose(0,2,1)
                    tmp = einsum('c,ld->lcd',r1,imds.Fov[kl])
                    Hr2[kl,kshift] += 2.*tmp*(kl==kd)
                    Hr2[kl,kd]     +=   -tmp.transpose(0,2,1)*(kl==kd)

        def mem_usage_voovk(nocc, nvir, nkpts):
            return nocc**2 * nvir**2 * nkpts * 16.
        array_size = [nkpts,nkpts]
        task_list = generate_max_task_list(array_size, mem_usage_voovk(nocc,nvir,nkpts), priority_list=[1,1])
        for kjrange, kbrange in mpi.work_stealing_partition(task_list):
            WvoovR1_jbX  = _cp(imds.WvoovR1[slice(*kjrange),slice(*kbrange),:])
            WovovRev_jbX = _cp(imds.WovovRev[slice(*kjrange),slice(*kbrange),:])

            for iterkj, kj in enumerate(range(*kjrange)):
                for iterkb, kb in enumerate(range(*kbrange)):
                    for iterkl, kl in enumerate(range(nkpts)):
                        kc = kconserv[kj,kb,kshift]
                        Hr2[kl,kc] += 2.*einsum('jcb,bljd->lcd',r2[kj,kc],WvoovR1_jbX[iterkj,iterkb,kl])
                        Hr2[kl,kc] -=    einsum('jcb,lbjd->lcd',r2[kj,kc],WovovRev_jbX[iterkj,iterkb,kl])
                        kc = kconserv[kl,kj,kb]
                        kd = kconserv[kshift,kc,kl]
                        Hr2[kl,kc] -=    einsum('jdb,bljc->lcd',r2[kj,kd],WvoovR1_jbX[iterkj,iterkb,kl])
                        Hr2[kl,kc] -=    einsum('jbd,lbjc->lcd',r2[kj,kb],WovovRev_jbX[iterkj,iterkb,kl])

        def mem_usage_vvvvk(nocc, nvir, nkpts):
            return nocc**0 * nvir**4 * nkpts * 16.
        array_size = [nkpts,nkpts]
        task_list = generate_max_task_list(array_size, mem_usage_vvvvk(nocc,nvir,nkpts),priority_list=[1,1])
        for karange, kbrange in mpi.work_stealing_partition(task_list):
            Wvvvv_abX = _cp(imds.Wvvvv[slice(*karange),slice(*kbrange),:])

            for iterka, ka in enumerate(range(*karange)):
                for iterkb, kb in enumerate(range(*kbrange)):
                    for iterkc, kc in enumerate(range(nkpts)):
                        kl = kconserv[ka,kshift,kb]
                        Hr2[kl,kc] += einsum('lab,abcd->lcd',r2[kl,ka],Wvvvv_abX[iterka,iterkb,kc])

        def mem_usage_oovvk(nocc, nvir, nkpts):
            return nocc**2 * nvir**2 * nkpts * 16.
        array_size = [nkpts,nkpts]
        task_list = generate_max_task_list(array_size, mem_usage_oovvk(nocc,nvir,nkpts),priority_list=[1,1])
        # TODO mpi.work_stealing_partition returns [[0,1]] for one dimension
        # and this doesn't work with the kjrange
        # Making term for Woovv contraction
        tmp_occ = numpy.zeros(nocc,dtype=t1.dtype)
        for kirange, kcrange in mpi.work_stealing_partition(task_list):
            for iterki, ki in enumerate(range(*kirange)):
                for iterkc, kc in enumerate(range(*kcrange)):
                    kb = kconserv[ki,kc,kshift]
                    tmp2 = unpack_tril(t2,nkpts,ki,kshift,kc,kconserv[ki,kc,kshift])
                    tmp_occ += einsum('ijcb,ibc->j',tmp2,r2[ki,kb])
        comm.Allreduce(MPI.IN_PLACE, tmp_occ, op=MPI.SUM)

        for kkrange, kfrange in mpi.work_stealing_partition(task_list):
            Woovv_kf = _cp(imds.Woovv[slice(*kkrange),kshift,slice(*kfrange)])
            for iterkk, kk in enumerate(range(*kkrange)):
                for iterkf, kf in enumerate(range(*kfrange)):
                    ke = kconserv[kshift,kf,kk]
                    tmp = einsum('kjfe,j->kef',Woovv_kf[iterkk,iterkf],tmp_occ)
                    Hr2[kk,ke] += -2.*tmp
                    Hr2[kk,kf] += tmp.transpose(0,2,1)

        comm.Allreduce(MPI.IN_PLACE, Hr2, op=MPI.SUM)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_size_ea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        size = nvir + nkpts ** 2 * nvir ** 2 * nocc
        return size

    mask_frozen_ea = mask_frozen_ea

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
        #r2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), vector.dtype)
        #index = nvir
        #for kj in range(nkpts):
        #    for ka in range(nkpts):
        #        for j in range(nocc):
        #            for a in range(nvir):
        #                for b in range(nvir):
        #                    r2[kj,ka,j,a,b] = vector[index]
        #                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        size = nvir + nkpts*nkpts*nocc*nvir*nvir

        vector = np.zeros((size), r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nkpts*nkpts*nocc*nvir*nvir)
        #index = nvir
        #for kj in range(nkpts):
        #    for ka in range(nkpts):
        #        for j in range(nocc):
        #            for a in range(nvir):
        #                for b in range(nvir):
        #                    vector[index] = r2[kj,ka,j,a,b]
        #                    index += 1
        return vector

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        kconserv = self.kconserv
        eris = self.eris
        kshift = self.kshift
        t2 = self.t2
        foo = eris.fock[:,:nocc,:nocc]
        fvv = eris.fock[:,nocc:,nocc:]

        e = []
        assert len(eaccsd_evecs.shape) == 2  # Done at a single k-point, kshift
        for _eval, _evec, _levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
            print(_eval, _evec.shape, _levec.shape)
            l1,l2 = self.vector_to_amplitudes_ea(_levec)
            r1,r2 = self.vector_to_amplitudes_ea(_evec)

            # Transposing the l2 operator
            # (equations were worked out for the transpose of these operators!)

            l2_t = numpy.zeros_like(l2)
            r2_t = numpy.zeros_like(r2)
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[kshift,ka,kj]
                    l2_t[kj,kb] = l2[kj,ka].transpose(0,2,1)
                    r2_t[kj,kb] = r2[kj,ka].transpose(0,2,1)

            # Normalization for the left eigenvector
            ldotr = numpy.dot(l1.ravel(),r1.ravel()) + numpy.dot(l2.ravel(),r2.ravel())
            l2 = (l2 + 2.*l2_t)/3.
            r2 = r2_t.copy()
            r2_t = None
            l2_t = None

            l1 /= ldotr
            l2 /= ldotr
            l1 = l1.reshape(-1)
            r1 = r1.reshape(-1)

            lijabc_tmp  = numpy.zeros((nocc,nocc,nvir,nvir,nvir),dtype=t2.dtype)
            rijabc_tmp  = numpy.zeros((nocc,nocc,nvir,nvir,nvir),dtype=t2.dtype)
            def mem_usage_oovvk(nocc, nvir, nkpts):
                return nocc**2 * nvir**2 * nkpts * 16
            array_size = [nkpts,nkpts]
            # FIXME: find a good blocksize for this. right now just makes the smallest blocksize possible
            task_list = generate_max_task_list(array_size,blk_mem_size=1e12,priority_list=[1,1])
            #nproc = comm.Get_size()
            #chunk_size = get_max_blocksize_from_mem(0.3e9, 2.*mem_usage_oovvk(nocc,nvir,nkpts),
            #                                        array_size, priority_list=[1,1])
            #max_chunk_size = tuple(nkpts//numpy.array([nproc,nproc]))
            #if max_chunk_size[0] < chunk_size[0]:
            #    chunk_size = max_chunk_size
            #task_list = generate_task_list(chunk_size,array_size)

            star_energy = numpy.array(0.0 + 1j*0.0)
            for kirange, kjrange in mpi.work_stealing_partition(task_list):
                #eris_oovv_jiX = _cp(eris.oovv[slice(*kjrange),slice(*kirange),range(nkpts)])
                eris_oovv_ijX = _cp(eris.oovv[slice(*kirange),slice(*kjrange),range(nkpts)])

                eris_ooov_jiX = _cp(eris.ooov[slice(*kjrange),slice(*kirange),range(nkpts)])
                eris_ooov_ijX = _cp(eris.ooov[slice(*kirange),slice(*kjrange),range(nkpts)])

                kclist = kpts_helper.get_kconserv3(self._scf.cell, self.kpts,
                                                   [range(*kirange),range(*kjrange),kshift,range(nkpts),range(nkpts)])

                for iterki, ki in enumerate(range(*kirange)):
                    for iterkj, kj in enumerate(range(*kjrange)):

                        # Starting the left amplitude equations

                        for iterka, ka in enumerate(range(nkpts)):
                            for iterkb, kb in enumerate(range(nkpts)):
                                if ka > kb:
                                    continue
                                kc = kclist[iterki,iterkj,ka,kb]
                                fac = (2.0 - float(ka==kb))

                                if ka == kb:
                                    if ki > kj:
                                        continue
                                    else:
                                        fac *= (2.0 - float(ki==kj))

                                lijabc_tmp *= 0.0
                                rijabc_tmp *= 0.0

                                #(a,b,c) -> (a,b,c)
                                if kc == kshift and kb == kconserv[ki,ka,kj]:
                                    tmp = -0.5*einsum('ijab,c->ijabc',eris_oovv_ijX[iterki,iterkj,ka],l1)
                                    lijabc_tmp += 8.*tmp

                                #(b,a,c) -> (a,b,c)
                                if kc == kshift and ka == kconserv[ki,kb,kj]:
                                    tmp = -0.5*einsum('ijba,c->ijabc',eris_oovv_ijX[iterki,iterkj,kb],l1)
                                    lijabc_tmp += -4.* tmp

                                #(c,b,a) -> (a,b,c)
                                if ka == kshift and kb == kconserv[ki,kc,kj]:
                                    tmp = -0.5*einsum('ijcb,a->ijabc',eris_oovv_ijX[iterki,iterkj,kc],l1)
                                    lijabc_tmp += -4.* tmp

                                #(a,c,b) -> (a,b,c)
                                if kb == kshift and kc == kconserv[ki,ka,kj]:
                                    tmp = -0.5*einsum('ijac,b->ijabc',eris_oovv_ijX[iterki,iterkj,ka],l1)
                                    lijabc_tmp += -4.* tmp

                                #(b,c,a) -> (a,b,c)
                                if kb == kshift and ka == kconserv[ki,kc,kj]:
                                    tmp = -0.5*einsum('ijca,b->ijabc',eris_oovv_ijX[iterki,iterkj,kc],l1)
                                    lijabc_tmp += 2.* tmp

                                #(c,a,b) -> (a,b,c)
                                if ka == kshift and kc == kconserv[ki,kb,kj]:
                                    tmp = -0.5*einsum('ijbc,a->ijabc',eris_oovv_ijX[iterki,iterkj,kb],l1)
                                    lijabc_tmp += 2.* tmp

                                # end of oovv terms

                                #(a,b,c) -> (a,b,c)
                                km = kconserv[ki,ka,kj]
                                tmp = einsum('jima,mbc->ijabc',eris_ooov_jiX[iterkj,iterki,km],2.*l2[km,kb] - l2[km,kc].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                km = kconserv[kj,kb,ki]
                                tmp = einsum('ijmb,mac->ijabc',eris_ooov_ijX[iterki,iterkj,km],2.*l2[km,ka] - l2[km,kc].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                #(b,a,c) -> (a,b,c)
                                km = kconserv[ki,kb,kj]
                                tmp = einsum('jimb,mac->ijabc',eris_ooov_jiX[iterkj,iterki,km],2.*l2[km,ka] - l2[km,kc].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                km = kconserv[kj,ka,ki]
                                tmp = einsum('ijma,mbc->ijabc',eris_ooov_ijX[iterki,iterkj,km],2.*l2[km,kb] - l2[km,kc].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                #(c,b,a) -> (a,b,c)
                                km = kconserv[ki,kc,kj]
                                tmp = einsum('jimc,mba->ijabc',eris_ooov_jiX[iterkj,iterki,km],2.*l2[km,kb] - l2[km,ka].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                km = kconserv[kj,kc,ki]
                                tmp = einsum('ijmc,mab->ijabc',eris_ooov_ijX[iterki,iterkj,km],2.*l2[km,ka] - l2[km,kb].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                # ovvv term 1

                                # ke = ks - kc + kj
                                # kb = ki + ke - ka
                                #    = ki + ks - kc + kj - ka
                                #    = ki + kj + ks - ka - kc

                                #(c,b,a) -> (a,b,c)
                                ke = kconserv[kshift,ka,ki]
                                tmp = -einsum('jebc,iae->ijabc',eris.ovvv[kj,ke,kb],2.*l2[ki,ka]-l2[ki,ke].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                ke = kconserv[kshift,kb,kj]
                                tmp = -einsum('ieac,jbe->ijabc',eris.ovvv[ki,ke,ka],2.*l2[kj,kb]-l2[kj,ke].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                #(b,c,a) -> (a,b,c)
                                ke = kconserv[kshift,kb,kj]
                                tmp = -einsum('ieca,jbe->ijabc',eris.ovvv[ki,ke,kc],2.*l2[kj,kb]-l2[kj,ke].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                ke = kconserv[kshift,kb,ki]
                                tmp = -einsum('jeac,ibe->ijabc',eris.ovvv[kj,ke,ka],2.*l2[ki,kb]-l2[ki,ke].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                #(c,a,b) -> (a,b,c)
                                ke = kconserv[kshift,ka,kj]
                                tmp = -einsum('iebc,jae->ijabc',eris.ovvv[ki,ke,kb],2.*l2[kj,ka]-l2[kj,ke].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                ke = kconserv[kshift,ka,ki]
                                tmp = -einsum('jecb,iae->ijabc',eris.ovvv[kj,ke,kc],2.*l2[ki,ka]-l2[ki,ke].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                # ovvv term 2

                                # ke = kshift + ki - ka
                                # kb = kj + ke - kc
                                #    = kj + ki + kshift - ka - kc

                                #(c,b,a) -> (a,b,c)
                                ke = kconserv[kshift,kc,ki]
                                tmp = -einsum('jeba,iec->ijabc',eris.ovvv[kj,ke,kb],2.*l2[ki,ke]-l2[ki,kc].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                ke = kconserv[kshift,kc,kj]
                                tmp = -einsum('ieab,jec->ijabc',eris.ovvv[ki,ke,ka],2.*l2[kj,ke]-l2[kj,kc].transpose(0,2,1))
                                lijabc_tmp += 2.* tmp

                                #(b,c,a) -> (a,b,c)
                                ke = kconserv[kshift,kc,ki]
                                tmp = -einsum('jeab,iec->ijabc',eris.ovvv[kj,ke,ka],2.*l2[ki,ke]-l2[ki,kc].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                ke = kconserv[kshift,ka,kj]
                                tmp = -einsum('iecb,jea->ijabc',eris.ovvv[ki,ke,kc],2.*l2[kj,ke]-l2[kj,ka].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                #(c,a,b) -> (a,b,c)
                                ke = kconserv[kshift,kb,ki]
                                tmp = -einsum('jeca,ieb->ijabc',eris.ovvv[kj,ke,kc],2.*l2[ki,ke]-l2[ki,kb].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp

                                ke = kconserv[kshift,kc,kj]
                                tmp = -einsum('ieba,jec->ijabc',eris.ovvv[ki,ke,kb],2.*l2[kj,ke]-l2[kj,kc].transpose(0,2,1))
                                lijabc_tmp += -1.* tmp


                                # Starting the right amplitude equations

                                # kf = ks
                                # ke = ki + kj - ka
                                # kc = kf + ke - kb
                                #    = ks + ki + kj - ka - kb
                                ke = kconserv[ki,ka,kj]
                                vvvv = _cp(eris.vvvv[kb,kc,ke])
                                tmp2 = einsum('bcef,f->bce',vvvv,r1)
                                tril = unpack_tril(t2,nkpts,ki,kj,ka,kconserv[ki,ka,kj])
                                tmp = - einsum('bce,ijae->ijabc',tmp2,tril)
                                rijabc_tmp += tmp

                                ke = kconserv[kj,kb,ki]
                                vvvv = _cp(eris.vvvv[ka,kc,ke])
                                tmp2 = einsum('acef,f->ace',vvvv,r1)
                                tril = unpack_tril(t2,nkpts,kj,ki,kb,kconserv[kj,kb,ki])
                                tmp = - einsum('ace,jibe->ijabc',tmp2,tril)
                                rijabc_tmp += tmp
                                ke = kshift
                                km = kconserv[ke,kc,kj]
                                ovov = _cp(eris.ovov[km,kc,kj])
                                tmp2 = einsum('mcje,e->mcj',ovov,r1)
                                tril = unpack_tril(t2,nkpts,ki,km,ka,kconserv[ki,ka,km])
                                tmp = einsum('mcj,imab->ijabc',tmp2,tril)
                                rijabc_tmp += tmp

                                ke = kshift
                                km = kconserv[ke,kc,ki]
                                ovov = _cp(eris.ovov[km,kc,ki])
                                tmp2 = einsum('mcie,e->mci',ovov,r1)
                                tril = unpack_tril(t2,nkpts,kj,km,kb,kconserv[kj,kb,km])
                                tmp = einsum('mci,jmba->ijabc',tmp2,tril)
                                rijabc_tmp += tmp

                                ke = kshift
                                km = kconserv[kc,ki,ka]
                                ovvo = _cp(eris.ovvo[km,kb,ke])
                                tmp2 = einsum('mbej,e->mbj',ovvo,r1)
                                tril = unpack_tril(t2,nkpts,ki,km,ka,kconserv[ki,ka,km])
                                tmp = einsum('mbj,imac->ijabc',tmp2,tril)
                                rijabc_tmp += tmp

                                ke = kshift
                                km = kconserv[kc,kj,kb]
                                ovvo = _cp(eris.ovvo[km,ka,ke])
                                tmp2 = einsum('maei,e->mai',ovvo,r1)
                                tril = unpack_tril(t2,nkpts,kj,km,kb,kconserv[kj,kb,km])
                                tmp = einsum('mai,jmbc->ijabc',tmp2,tril)
                                rijabc_tmp += tmp

                                ks = kshift
                                km = kconserv[ki,ka,kj]
                                tmp = einsum('maji,mbc->ijabc',eris.ovoo[km,ka,kj],r2[km,kb])
                                rijabc_tmp += tmp

                                ks = kshift
                                km = kconserv[kj,kb,ki]
                                tmp = einsum('mbij,mac->ijabc',eris.ovoo[km,kb,ki],r2[km,ka])
                                rijabc_tmp += tmp

                                ks = kshift
                                ke = kconserv[ks,ka,ki]
                                tmp = -einsum('ejcb,iae->ijabc',eris.vovv[ke,kj,kc].conj(),r2[ki,ka])
                                rijabc_tmp += tmp

                                ks = kshift
                                ke = kconserv[ks,kb,kj]
                                tmp = -einsum('eica,jbe->ijabc',eris.vovv[ke,ki,kc].conj(),r2[kj,kb])
                                rijabc_tmp += tmp

                                ks = kshift
                                ke = kconserv[ks,kc,kj]
                                tmp = -einsum('eiba,jec->ijabc',eris.vovv[ke,ki,kb].conj(),r2[kj,ke])
                                rijabc_tmp += tmp

                                ks = kshift
                                ke = kconserv[ks,kc,ki]
                                tmp = -einsum('ejab,iec->ijabc',eris.vovv[ke,kj,ka].conj(),r2[ki,ke])
                                rijabc_tmp += tmp

                                eia = numpy.diagonal(foo[ki]).reshape(-1,1) - numpy.diagonal(fvv[ka])
                                eia += _eval
                                ejb = numpy.diagonal(foo[kj]).reshape(-1,1) - numpy.diagonal(fvv[kb])
                                eijab = pyscf.lib.direct_sum('ia,jb->ijab',eia,ejb)

                                kc = kclist[iterki,iterkj,ka,kb]
                                eijabc = pyscf.lib.direct_sum('ijab,c->ijabc',eijab,-numpy.diagonal(fvv[kc]))
                                eijabc = 1./eijabc

                                star_energy += fac*0.5*einsum('ijabc,ijabc,ijabc',lijabc_tmp,rijabc_tmp,eijabc)

            comm.Allreduce(MPI.IN_PLACE, star_energy, op=MPI.SUM)
            if rank == 0:
                logger.info(self, 'EOM-EACCSD* delta energy = %.15g (Imag = %.15g)',
                            star_energy.real,star_energy.imag)
            e.append(star_energy+_eval)

        return np.array(e)

    def amplitudes_to_vector(self, t1, t2):
        return np.hstack((t1.ravel(), t2.ravel()))

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        nov = nkpts*nocc*nvir
        t1 = vec[:nov].reshape(nkpts,nocc,nvir)
        t2 = vec[nov:].reshape(nkpts*(nkpts+1)//2,nkpts,nocc,nocc,nvir,nvir)
        return t1, t2

    @property
    def imds(self):
        if self.__imds__ is None:
            self.__imds__ = _IMDS(self)
        return self.__imds__

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        cput0 = (logger.process_clock(), logger.perf_counter())
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        if mo_coeff is None:
            # If mo_coeff is not canonical orbital
            # TODO does this work for k-points? changed to conjugate.
            raise NotImplementedError
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        with lib.temporary_env(cc._scf, exxdiv=None):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cell, dm)
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
        self.fock = comm.bcast(self.fock, root=0)  # Ensure all processes have same fock

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
        # Add HFX correction in the self.mo_energy to improve convergence in
        # CCSD iteration. It is useful for the 2D systems since their occupied and
        # the virtual orbital energies may overlap which may lead to numerical
        # issue in the CCSD iterations.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the correct value.
        madelung = tools.madelung(cell, kpts)
        self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)

        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.kconserv
        khelper = cc.khelper

        log = logger.Logger(cc.stdout, cc.verbose)
        #mem_now = pyscf.lib.current_memory()[0]
        #if (method == 'incore' and (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
        if False:
            log.info('using incore ERI storage')
            eri = numpy.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo), dtype=numpy.complex128)
            unique_klist = list(khelper.symm_map.keys())
            nUnique_klist = len(unique_klist)

            # Looping over unique list of k-vectors
            loader = mpi_load_balancer.load_balancer(BLKSIZE=(nkpts,))
            loader.set_ranges((range(nUnique_klist),))

            good2go = True
            while (good2go):
                good2go, data = loader.slave_set()
                if good2go is False:
                    break
                index = 0
                block = data[index]
                ranges = loader.outblocks[index][block]
                for indices in ranges:
                    kp, kq, kr = unique_klist[indices]
                    ks = kconserv[kp,kq,kr]
                    eri_kpt = fao2mo((mo_coeff[kp],mo_coeff[kq],mo_coeff[kr],mo_coeff[ks]),
                                     (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]))
                    eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                    eri[kp,kq,kr] = eri_kpt.copy()
                    loader.slave_finished()

            comm.Barrier()
            comm.Allreduce(MPI.IN_PLACE, eri, op=MPI.SUM)
            comm.Barrier()

            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ikp, ikq, ikr = khelper.get_irrVec(kp,kq,kr)
                        irr_eri = eri[ikp,ikq,ikr]
                        eri[kp,kq,kr] = khelper.transform_irr2full(irr_eri,kp,kq,kr)
            comm.Barrier()

            # Chemist -> physics notation
            eri = eri.transpose(0,2,1,3,5,4,6)

            self.dtype = eri.dtype
            self.oooo = eri[:,:,:,:nocc,:nocc,:nocc,:nocc].copy() / nkpts
            self.ooov = eri[:,:,:,:nocc,:nocc,:nocc,nocc:].copy() / nkpts
            self.ovoo = eri[:,:,:,:nocc,nocc:,:nocc,:nocc].copy() / nkpts
            self.oovv = eri[:,:,:,:nocc,:nocc,nocc:,nocc:].copy() / nkpts
            self.ovov = eri[:,:,:,:nocc,nocc:,:nocc,nocc:].copy() / nkpts
            self.ovvv = eri[:,:,:,:nocc,nocc:,nocc:,nocc:].copy() / nkpts
            self.vvvv = eri[:,:,:,nocc:,nocc:,nocc:,nocc:].copy() / nkpts
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
        else:
            _tmpfile1_name = None
            if rank == 0:
                _tmpfile1_name = "eris1.hdf5"
            _tmpfile1_name = comm.bcast(_tmpfile1_name, root=0)

            read_feri=False
            if rank == 0:
                if os.path.isfile(_tmpfile1_name):
                    read_feri=True
            read_feri = comm.bcast(read_feri,root=0)

            if read_feri is True:
                self.feri1 = h5py.File(_tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
                self.oooo  = self.feri1['oooo']
                self.ooov  = self.feri1['ooov']
                self.ovoo  = self.feri1['ovoo']
                self.oovv  = self.feri1['oovv']
                self.ovov  = self.feri1['ovov']
                self.ovvo  = self.feri1['ovvo']
                self.voov  = self.feri1['voov']
                self.ovvv  = self.feri1['ovvv']
                self.vovv  = self.feri1['vovv']
                self.vvvv  = self.feri1['vvvv']

                self.ovovL1  = self.feri1['ovovL1']
                self.ooovL1  = self.feri1['ooovL1']
                #self.ovvvL1  = self.feri1['ovvvL1']
                self.voovR1  = self.feri1['voovR1']
                self.ooovR1  = self.feri1['ooovR1']
                self.vovvR1  = self.feri1['vovvR1']
                self.vovvL1  = self.feri1['vovvL1']
                self.ovovRev  = self.feri1['ovovRev']
                self.ooovRev  = self.feri1['ooovRev']
                self.ovvvRev  = self.feri1['ovvvRev']

                log.warn('using oovv integrals in memory')
                new_oovv = numpy.empty( (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
                for kp in range(nkpts):
                    for kq in range(nkpts):
                        for kr in range(nkpts):
                            new_oovv[kp,kq,kr] = self.oovv[kp,kq,kr].copy()
                self.oovv = new_oovv

                #print("lower triangular oovv")
                #self.triu_oovv = numpy.empty( ((nkpts*(nkpts+1))/2,nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff.dtype)
                #triu_indices = [list(x) for x in numpy.triu_indices(nkpts)]
                #self.triu_oovv = self.oovv[triu_indices]
                return
            comm.Barrier()

            self.feri1 = h5py.File(_tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

            ds_type = dtype

            self.oooo = self.feri1.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.ovoo = self.feri1.create_dataset('ovoo', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=ds_type)

            self.ovov = self.feri1.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ovvo = self.feri1.create_dataset('ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
            self.voov = self.feri1.create_dataset('voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
            self.ovvv = self.feri1.create_dataset('ovvv', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)
            self.vovv = self.feri1.create_dataset('vovv', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)

            self.ovovL1 = self.feri1.create_dataset('ovovL1', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ooovL1 = self.feri1.create_dataset('ooovL1', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            #self.ovvvL1 = self.feri1.create_dataset('ovvvL1', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)
            self.ovovRev = self.feri1.create_dataset('ovovRev', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
            self.ooovRev = self.feri1.create_dataset('ooovRev', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.ovvvRev = self.feri1.create_dataset('ovvvRev', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=ds_type)

            self.voovR1 = self.feri1.create_dataset('voovR1', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
            self.ooovR1 = self.feri1.create_dataset('ooovR1', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
            self.vovvR1 = self.feri1.create_dataset('vovvR1', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)
            self.vovvL1 = self.feri1.create_dataset('vovvL1', (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)

            mem = 0.5e9
            pre = 1.*nocc*nocc*nmo*nmo*nkpts*16
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)

            BLKSIZE = (1,nkpts_blksize,nkpts,)
            if rank == 0:
                log.info("ERI oopq blksize = (%3d %3d %3d)" % BLKSIZE)
            loader = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            def get_norbitals_from_string(string):
                '''Converts a string representing a block of integrals to number of orbitals in that block

                A list of conversions is as follows:
                    'p'    - number of molecule orbitals
                    'o'    - number of occupied orbitals
                    'v'    - number of virtual orbitals
                    'k'    - number of k-points
                '''
                if len(string) == 0:
                    return 0

                chars, counts = np.unique(list(string), return_counts=True)
                norbitals = 1.
                for ic, c in enumerate(chars):
                    if c == 'p':
                        norbitals *= nmo
                    elif c == 'o':
                        norbitals *= nocc
                    elif c == 'v':
                        norbitals *= nvir
                    elif c == 'k':
                        norbitals *= nkpts
                    else:
                        raise ValueError('String %s not allowed.')

                return norbitals

            #array_size = [nkpts, nkpts, nkpts]
            #task_list = generate_max_task_list(array_size,
            #                blk_mem_size=1e9*2*16*get_norbitals_from_string('kkoopp'),
            #                priority_list=[3,2,1])
            #                        for kirange, kjrange in mpi.work_stealing_partition(task_list):

            tmp_block_shape = BLKSIZE + (nocc,nocc,nmo,nmo)
            tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)
            cput1 = logger.process_clock(), logger.perf_counter()
            good2go = True
            while (good2go):
                good2go, data = loader.slave_set()
                if good2go is False:
                    break
                ranges0, ranges1, ranges2 = loader.get_blocks_from_data(data)
                #rslice = [slice(0,max(x)-min(x)) for x in (ranges0,ranges1,ranges2)]
                for kp in ranges0:
                    for kq in ranges2:
                        for kr in ranges1:
                            ks = kconserv[kp,kq,kr]
                            orbo_p = mo_coeff[kp][:,:nocc]
                            orbo_r = mo_coeff[kr][:,:nocc]
                            eri_kpt = fao2mo((orbo_p,mo_coeff[kq][:,:],orbo_r,mo_coeff[ks][:,:]),
                                             (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]))
                            eri_kpt = eri_kpt.reshape(nocc,nmo,nocc,nmo)
                            eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
                            tmp_block[kp-ranges0[0],kr-ranges1[0],kq-ranges2[0]] = eri_kpt

                self.oooo    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ooov    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.ooovL1  [min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(1,2,0,3,4,5,6)
                self.ooovR1  [min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,0,1,3,4,5,6)
                self.ooovRev [min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,1,0,3,4,5,6)
                self.oovv    [min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]

                loader.slave_finished()

            comm.Barrier()
            cput1 = log.timer_debug1('transforming oopq', *cput1)

            mem = 0.5e9
            pre = 1.*nocc*nvir*nmo*nmo*nkpts*16
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nkpts)

            BLKSIZE = (1,nkpts_blksize,nkpts,)
            if rank == 0:
                log.info("ERI ovpq blksize = (%3d %3d %3d)" % BLKSIZE)
            loader1 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader1.set_ranges((range(nkpts),range(nkpts),range(nkpts),))

            tmp_block_shape = BLKSIZE + (nocc,nvir,nmo,nmo)
            tmp_block  = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

            cput1 = logger.process_clock(), logger.perf_counter()
            good2go = True
            while (good2go):
                good2go, data = loader1.slave_set()
                if good2go is False:
                    break
                ranges0, ranges1, ranges2 = loader1.get_blocks_from_data(data)
                #rslice = [slice(0,len(x)) for x in (ranges0,ranges1,ranges2)]
                for kp in ranges0:
                    for kq in ranges2:
                        for kr in ranges1:
                            ks = kconserv[kp,kq,kr]
                            orbo_p = mo_coeff[kp][:,:nocc]
                            orbv_r = mo_coeff[kr][:,nocc:]
                            eri_kpt = fao2mo((orbo_p,mo_coeff[kq][:,:],orbv_r,mo_coeff[ks][:,:]),
                                             (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]))
                            eri_kpt = eri_kpt.reshape(nocc,nmo,nvir,nmo)
                            eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts
                            tmp_block[kp-ranges0[0],kr-ranges1[0],kq-ranges2[0]] = eri_kpt
                            self.voov[kr,kp,ks] = eri_kpt.transpose(1,0,3,2)[:,:,:nocc,nocc:]
                            self.voovR1[ks,kr,kp] = eri_kpt.transpose(1,0,3,2)[:,:,:nocc,nocc:]
                            self.vovv[kr,kp,ks] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]
                            self.vovvR1[ks,kr,kp] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]
                            self.vovvL1[kp,ks,kr] = eri_kpt.transpose(1,0,3,2)[:,:,nocc:,nocc:]

                self.ovoo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,:nocc]
                self.ovov[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:]
                self.ovovL1[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(1,2,0,3,4,5,6)
                self.ovovRev[min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,:nocc,nocc:].transpose(2,1,0,3,4,5,6)
                self.ovvo[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,:nocc]
                self.ovvv[min(ranges0):max(ranges0)+1,min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:]
                #self.ovvvL1[min(ranges1):max(ranges1)+1,min(ranges2):max(ranges2)+1,min(ranges0):max(ranges0)+1] = \
                #        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(1,2,0,3,4,5,6)
                self.ovvvRev[min(ranges2):max(ranges2)+1,min(ranges1):max(ranges1)+1,min(ranges0):max(ranges0)+1] = \
                        tmp_block[:len(ranges0),:len(ranges1),:len(ranges2),:,:,nocc:,nocc:].transpose(2,1,0,3,4,5,6)
                loader1.slave_finished()

            comm.Barrier()
            cput1 = log.timer_debug1('transforming ovpq', *cput1)

            # Here we can exploit the full 4-permutational symmetry
            # for 'vvvv' unlike in the cases above

            mem = 0.5e9
            pre = 1.*nvir*nvir*nvir*nvir*16
            unique_klist = list(khelper.symm_map.keys())
            nUnique_klist = len(unique_klist)
            nkpts_blksize = min(max(int(numpy.floor(mem/pre)),1),nUnique_klist)

            BLKSIZE = (nkpts_blksize,)
            if rank == 0:
                log.info("ERI vvvv blksize = %3d" % nkpts_blksize)
            loader2 = mpi_load_balancer.load_balancer(BLKSIZE=BLKSIZE)
            loader2.set_ranges((range(nUnique_klist),))

            tmp_block_shape = BLKSIZE + (nvir,nvir,nvir,nvir)
            tmp_block = numpy.empty(shape=tmp_block_shape,dtype=ds_type)

            good2go = True
            while (good2go):
                good2go, data = loader2.slave_set()
                if good2go is False:
                    break
                ranges = loader2.get_blocks_from_data(data)
                chkpts = [int(numpy.ceil(nUnique_klist/10))*i for i in range(10)]
                for indices in ranges:
                    if indices in chkpts:
                        log.info("vvvv transform is %4.2f percent complete" % (1.*indices/nUnique_klist*100))
                    kp, kq, kr = unique_klist[indices]
                    ks = kconserv[kp,kq,kr]
                    orbv_p = mo_coeff[kp][:,nocc:]
                    orbv_q = mo_coeff[kq][:,nocc:]
                    orbv_r = mo_coeff[kr][:,nocc:]
                    orbv_s = mo_coeff[ks][:,nocc:]
                    eri_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
                                     (cc.kpts[kp],cc.kpts[kq],cc.kpts[kr],cc.kpts[ks]))
                    eri_kpt = eri_kpt.reshape(nvir,nvir,nvir,nvir)
                    eri_kpt = eri_kpt.transpose(0,2,1,3) / nkpts

                    self.vvvv[kp,kr,kq] = eri_kpt.copy()
                    # Store symmetric permutations
                    self.vvvv[kr,kp,ks] = eri_kpt.transpose(1,0,3,2).copy()
                    self.vvvv[kq,ks,kp] = eri_kpt.transpose(2,3,0,1).conj().copy()
                    self.vvvv[ks,kq,kr] = eri_kpt.transpose(3,2,1,0).conj().copy()
                loader2.slave_finished()

            comm.Barrier()
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

            self.feri1.close()
            self.feri1 = h5py.File(_tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)
            self.oooo  = self.feri1['oooo']
            self.ooov  = self.feri1['ooov']
            self.ovoo  = self.feri1['ovoo']
            self.oovv  = self.feri1['oovv']
            self.ovov  = self.feri1['ovov']
            self.ovvo  = self.feri1['ovvo']
            self.voov  = self.feri1['voov']
            self.ovvv  = self.feri1['ovvv']
            self.vovv  = self.feri1['vovv']
            self.vvvv  = self.feri1['vvvv']

            self.ovovL1  = self.feri1['ovovL1']
            self.ooovL1  = self.feri1['ooovL1']
            #self.ovvvL1  = self.feri1['ovvvL1']
            self.voovR1  = self.feri1['voovR1']
            self.ooovR1  = self.feri1['ooovR1']
            self.vovvR1  = self.feri1['vovvR1']
            self.vovvL1  = self.feri1['vovvL1']
            self.ovovRev  = self.feri1['ovovRev']
            self.ooovRev  = self.feri1['ooovRev']
            self.ovvvRev  = self.feri1['ovvvRev']

            log.warn('using oovv integrals in memory')
            new_oovv = numpy.empty( (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        new_oovv[kp,kq,kr] = self.oovv[kp,kq,kr].copy()
            self.oovv = new_oovv

        log.timer('CCSD integral transformation', *cput0)

    def __del__(self):
        if getattr(self, 'feri1', None):
            #for key in self.feri1.keys(): del (self.feri1[key])
            self.feri1.close()

CCSD = RCCSD

class _IMDS:
    def __init__(self, cc):
        return

    def make_ip(self,cc):
        #cc = self.cc
        t1,t2,eris = cc.t1, cc.t2, cc.eris
        nkpts,nocc,nvir = t1.shape

        if not getattr(self, 'fint1', None):
            self.fint1 = None

        tmpfile1_name = "eom_intermediates_IP.hdf5"
        self.fint1 = h5py.File(tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        ds_type = t2.dtype

        self.Wooov  = self.fint1.create_dataset('Wooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
        self.Woooo  = self.fint1.create_dataset('Woooo',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
        self.WooooS = self.fint1.create_dataset('WooooS',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=ds_type)
        self.W1voov = self.fint1.create_dataset('W1voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.W2voov = self.fint1.create_dataset('W2voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.Wvoov  = self.fint1.create_dataset('Wvoov',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        #self.W1ovvo = self.fint1.create_dataset('W1ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.W2ovvo = self.fint1.create_dataset('W2ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.Wovvo  = self.fint1.create_dataset('Wovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        self.W1ovov = self.fint1.create_dataset('W1ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.W2ovov = self.fint1.create_dataset('W2ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.Wovov  = self.fint1.create_dataset('Wovov',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.Wovoo  = self.fint1.create_dataset('Wovoo',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=ds_type)

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)

        #
        # Order matters here for array creation
        self.Wooov = imdk.Wooov(cc,t1,t2,eris,self.fint1)

        self.W1voov = imdk.W1voov(cc,t1,t2,eris,self.fint1)
        self.W2voov = imdk.W2voov(cc,t1,t2,eris,self.fint1)
        self.Wvoov = imdk.Wvoov(cc,t1,t2,eris,self.fint1)

        #self.W1ovvo = imdk.W1ovvo(cc,t1,t2,eris,self.fint1)
        #self.W2ovvo = imdk.W2ovvo(cc,t1,t2,eris,self.fint1)
        #self.Wovvo = imdk.Wovvo(cc,t1,t2,eris,self.fint1)

        self.Woooo = imdk.Woooo(cc,t1,t2,eris,self.fint1)
        self.WooooS = imdk.WooooS(cc,t1,t2,eris,self.fint1)

        self.W1ovov = imdk.W1ovov(cc,t1,t2,eris,self.fint1)
        self.W2ovov = imdk.W2ovov(cc,t1,t2,eris,self.fint1)
        self.Wovov  = imdk.Wovov(cc,t1,t2,eris,self.fint1)

        self.Woovv = eris.oovv

        self.Wovoo = imdk.Wovoo(cc,t1,t2,eris,self.fint1)

        self.fint1.close()
        self.fint1 = h5py.File(tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        self.Wooov  = self.fint1['Wooov' ]
        self.Woooo  = self.fint1['Woooo' ]
        self.WooooS = self.fint1['WooooS' ]
        #self.W1ovvo = self.fint1['W1ovvo']
        #self.W2ovvo = self.fint1['W2ovvo']
        #self.Wovvo  = self.fint1['Wovvo' ]
        self.W1voov = self.fint1['W1voov']
        self.W2voov = self.fint1['W2voov']
        self.Wvoov  = self.fint1['Wvoov' ]
        self.W1ovov = self.fint1['W1ovov']
        self.W2ovov = self.fint1['W2ovov']
        self.Wovov  = self.fint1['Wovov' ]
        self.Wovoo  = self.fint1['Wovoo' ]

    def close_ip(self,cc):
        self.fint1.close()

    def __del__(self):
        if getattr(self, 'fint1', None):
            #for key in self.feri1.keys(): del (self.feri1[key])
            self.fint1.close()
        if getattr(self, 'fint2', None):
            #for key in self.feri1.keys(): del (self.feri1[key])
            self.fint2.close()

    def make_ea(self,cc):
        t1,t2,eris = cc.t1, cc.t2, cc.eris
        nkpts,nocc,nvir = t1.shape

        if not getattr(self, 'fint2', None):
            self.fint2 = None

        tmpfile1_name = "eom_intermediates_EA.hdf5"
        self.fint2 = h5py.File(tmpfile1_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        ds_type = t2.dtype

        self.Wooov  = self.fint2.create_dataset('Wooov',  (nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=ds_type)
        self.Wvovv  = self.fint2.create_dataset('Wvovv',  (nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=ds_type)

        #self.W1ovvo = self.fint2.create_dataset('W1ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.W2ovvo = self.fint2.create_dataset('W2ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)
        #self.Wovvo  = self.fint2.create_dataset('Wovvo',  (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=ds_type)

        self.W1voov = self.fint2.create_dataset('W1voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.W2voov = self.fint2.create_dataset('W2voov', (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
#
        #self.Wvoov  = self.fint2.create_dataset('Wvoov',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.WvoovR1 = self.fint2.create_dataset('WvoovR1',  (nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=ds_type)
        self.Wvvvv   = self.fint2.create_dataset('Wvvvv',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=ds_type)
        self.W1ovov  = self.fint2.create_dataset('W1ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.W2ovov  = self.fint2.create_dataset('W2ovov',  (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
#
        #self.Wovov  = self.fint2.create_dataset('Wovov',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
        self.WovovRev  = self.fint2.create_dataset('WovovRev',   (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=ds_type)
#
        #self.Wvvvo  = self.fint2.create_dataset('Wvvvo',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=ds_type)
        self.WvvvoR1  = self.fint2.create_dataset('WvvvoR1',  (nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=ds_type)

        self.Lvv = imdk.Lvv(cc,t1,t2,eris)
        self.Loo = imdk.Loo(cc,t1,t2,eris)
        self.Fov = imdk.cc_Fov(cc,t1,t2,eris)

        #
        # Order matters here for array creation
        self.Wooov = imdk.Wooov(cc,t1,t2,eris,self.fint2)

        self.Wvovv = imdk.Wvovv(cc,t1,t2,eris,self.fint2)

        self.W1voov = imdk.W1voov(cc,t1,t2,eris,self.fint2)
        self.W2voov = imdk.W2voov(cc,t1,t2,eris,self.fint2)
#
        #self.Wvoov = imdk.Wvoov(cc,t1,t2,eris,self.fint2)
        self.WvoovR1 = imdk.WvoovR1(cc,t1,t2,eris,self.fint2)

        #self.W1ovvo = imdk.W1ovvo(cc,t1,t2,eris,self.fint2)
        #self.W2ovvo = imdk.W2ovvo(cc,t1,t2,eris,self.fint2)
        #self.Wovvo = imdk.Wovvo(cc,t1,t2,eris,self.fint2)

        self.Wvvvv = imdk.Wvvvv(cc,t1,t2,eris,self.fint2)

        self.Woovv = eris.oovv

        self.W1ovov = imdk.W1ovov(cc,t1,t2,eris,self.fint2)
        self.W2ovov = imdk.W2ovov(cc,t1,t2,eris,self.fint2)
#
        #self.Wovov  = imdk.Wovov(cc,t1,t2,eris,self.fint2)
        self.WovovRev  = imdk.WovovRev(cc,t1,t2,eris,self.fint2)

#
        #self.Wvvvo = imdk.Wvvvo(cc,t1,t2,eris,self.fint2)
        self.WvvvoR1 = imdk.WvvvoR1(cc,t1,t2,eris,self.fint2)

        self.fint2.close()
        self.fint2 = h5py.File(tmpfile1_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        self.Wooov  = self.fint2['Wooov' ]
        self.Wvovv  = self.fint2['Wvovv' ]
        #self.W1ovvo = self.fint2['W1ovvo']
        #self.W2ovvo = self.fint2['W2ovvo']
        #self.Wovvo  = self.fint2['Wovvo' ]
        self.W1voov = self.fint2['W1voov']
        self.W2voov = self.fint2['W2voov']
#
        #self.Wvoov  = self.fint2['Wvoov' ]
        self.WvoovR1  = self.fint2['WvoovR1' ]
        self.Wvvvv  = self.fint2['Wvvvv' ]
        self.W1ovov = self.fint2['W1ovov']
        self.W2ovov = self.fint2['W2ovov']
#
        #self.Wovov  = self.fint2['Wovov' ]
        self.WovovRev  = self.fint2['WovovRev' ]
#
        #self.Wvvvo  = self.fint2['Wvvvo' ]
        self.WvvvoR1  = self.fint2['WvvvoR1' ]

def _cp(a):
    return np.array(a, copy=False, order='C')

