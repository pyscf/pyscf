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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
RHF-CCSD(T) for real integrals
'''


import ctypes
import numpy
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.cc import _ccsd

# t3 as ijkabc

# JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    cpu1 = cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    name = mycc.__class__.__name__
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dtype = numpy.result_type(t1, t2, eris.ovoo.dtype)
    if mycc.incore_complete:
        ftmp = None
        eris_vvop = numpy.zeros((nvir,nvir,nocc,nmo), dtype)
    else:
        ftmp = lib.H5TmpFile()
        eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), dtype)

    orbsym = _sort_eri(mycc, eris, nocc, nvir, eris_vvop, log)

    mo_energy, t1T, t2T, vooo, fvo, restore_t2_inplace = \
            _sort_t2_vooo_(mycc, orbsym, t1, t2, eris)
    cpu1 = log.timer_debug1(f'{name}(T) sort_eri', *cpu1)

    cpu2 = list(cpu1)
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    if dtype == numpy.complex128:
        drv = _ccsd.libcc.CCsd_t_zcontract
    else:
        drv = _ccsd.libcc.CCsd_t_contract
    et_sum = numpy.zeros(1, dtype=dtype)
    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)

    # The rest 20% memory for cache b
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    bufsize = (max_memory*.5e6/8-nocc**3*3*lib.num_threads())/(nocc*nmo)  #*.5 for async_io
    bufsize *= .5  #*.5 upper triangular part is loaded
    bufsize *= .8  #*.8 for [a0:a1]/[b0:b1] partition
    bufsize = max(8, bufsize)
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    with lib.call_in_background(contract, sync=not mycc.async_io) as async_contract:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvir, bufsize))):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            async_contract(a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                            cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/8):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                async_contract(a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                                cache_row_b,cache_col_b))

    t2 = restore_t2_inplace(t2T)
    et_sum *= 2
    if abs(et_sum[0].imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part of %s(T) energy was found %s',
                    name, et_sum[0])
    et = et_sum[0].real
    log.timer(f'{name}(T)', *cpu0)
    log.note('%s(T) correction = %.15g', name, et)
    return et

def _sort_eri(mycc, eris, nocc, nvir, vvop, log):
    cpu1 = (logger.process_clock(), logger.perf_counter())
    mol = mycc.mol
    nmo = nocc + nvir

    if mol.symmetry:
        ovlp = mycc._scf.get_ovlp()
        orbsym = symm.addons.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                            eris.mo_coeff, s=ovlp, check=False)
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32) % 10
    else:
        orbsym = numpy.zeros(nmo, dtype=numpy.int32)

    o_sorted = _irrep_argsort(orbsym[:nocc])
    v_sorted = _irrep_argsort(orbsym[nocc:])
    vrank = numpy.argsort(v_sorted)

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.9)
    blksize = min(nvir, max(16, int(max_memory*1e6/8/(nvir*nocc*nmo))))
    log.debug1('_sort_eri max_memory %g  blksize %d', max_memory, blksize)
    dtype = vvop.dtype
    with lib.call_in_background(vvop.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((nocc,nmo,nvir), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvir, blksize):
            ovov = numpy.asarray(eris.ovov[:,j0:j1])
            #ovvv = numpy.asarray(eris.ovvv[:,j0:j1])
            ovvv = eris.get_ovvv(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                oov = ovov[o_sorted,j-j0]
                ovv = ovvv[o_sorted,j-j0]
                #if ovv.ndim == 2:
                #    ovv = lib.unpack_tril(ovv, out=buf)
                bufopv[:,:nocc,:] = oov[:,o_sorted][:,:,v_sorted].conj()
                bufopv[:,nocc:,:] = ovv[:,v_sorted][:,:,v_sorted].conj()
                save(vrank[j], bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    return orbsym

def _sort_t2_vooo_(mycc, orbsym, t1, t2, eris):
    assert (t2.flags.c_contiguous)
    vooo = numpy.asarray(eris.ovoo).transpose(1,0,2,3).conj().copy()
    nocc, nvir = t1.shape
    if mycc.mol.symmetry:
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32)
        o_sorted = _irrep_argsort(orbsym[:nocc])
        v_sorted = _irrep_argsort(orbsym[nocc:])
        mo_energy = eris.mo_energy
        mo_energy = numpy.hstack((mo_energy[:nocc][o_sorted],
                                  mo_energy[nocc:][v_sorted]))
        t1T = numpy.asarray(t1.T[v_sorted][:,o_sorted], order='C')
        fvo = eris.fock[nocc:,:nocc]
        fvo = numpy.asarray(fvo[v_sorted][:,o_sorted], order='C')

        o_sym = orbsym[o_sorted]
        oo_sym = (o_sym[:,None] ^ o_sym).ravel()
        oo_sorted = _irrep_argsort(oo_sym)
        #:vooo = eris.ovoo.transpose(1,0,2,3)
        #:vooo = vooo[v_sorted][:,o_sorted][:,:,o_sorted][:,:,:,o_sorted]
        #:vooo = vooo.reshape(nvir,-1,nocc)[:,oo_sorted]
        oo_idx = numpy.arange(nocc**2).reshape(nocc,nocc)[o_sorted][:,o_sorted]
        oo_idx = oo_idx.ravel()[oo_sorted]
        oo_idx = (oo_idx[:,None]*nocc+o_sorted).ravel()
        vooo = lib.take_2d(vooo.reshape(nvir,-1), v_sorted, oo_idx)
        vooo = vooo.reshape(nvir,nocc,nocc,nocc)

        #:t2T = t2.transpose(2,3,1,0)
        #:t2T = ref_t2T[v_sorted][:,v_sorted][:,:,o_sorted][:,:,:,o_sorted]
        #:t2T = ref_t2T.reshape(nvir,nvir,-1)[:,:,oo_sorted]
        t2T = lib.transpose(t2.reshape(nocc**2,-1))
        oo_idx = numpy.arange(nocc**2).reshape(nocc,nocc).T[o_sorted][:,o_sorted]
        oo_idx = oo_idx.ravel()[oo_sorted]
        vv_idx = (v_sorted[:,None]*nvir+v_sorted).ravel()
        t2T = lib.take_2d(t2T.reshape(nvir**2,nocc**2), vv_idx, oo_idx, out=t2)
        t2T = t2T.reshape(nvir,nvir,nocc,nocc)
        def restore_t2_inplace(t2T):
            tmp = numpy.zeros((nvir**2,nocc**2), dtype=t2T.dtype)
            lib.takebak_2d(tmp, t2T.reshape(nvir**2,nocc**2), vv_idx, oo_idx)
            t2 = lib.transpose(tmp.reshape(nvir**2,nocc**2), out=t2T)
            return t2.reshape(nocc,nocc,nvir,nvir)
    else:
        fvo = eris.fock[nocc:,:nocc].copy()
        t1T = t1.T.copy()
        t2T = lib.transpose(t2.reshape(nocc**2,nvir**2))
        t2T = lib.transpose(t2T.reshape(nvir**2,nocc,nocc), axes=(0,2,1), out=t2)
        mo_energy = numpy.asarray(eris.mo_energy, order='C')
        def restore_t2_inplace(t2T):
            tmp = lib.transpose(t2T.reshape(nvir**2,nocc,nocc), axes=(0,2,1))
            t2 = lib.transpose(tmp.reshape(nvir**2,nocc**2), out=t2T)
            return t2.reshape(nocc,nocc,nvir,nvir)
    t2T = t2T.reshape(nvir,nvir,nocc,nocc)
    return mo_energy, t1T, t2T, vooo, fvo, restore_t2_inplace

def _irrep_argsort(orbsym):
    return numpy.hstack([numpy.where(orbsym == i)[0] for i in range(8)])


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

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
    print(e3a - -0.0033300722704016289)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.symmetry = True

    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a - -0.003060022611584471)
