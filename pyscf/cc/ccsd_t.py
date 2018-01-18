#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import gc
import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.cc import _ccsd

'''
CCSD(T)
'''

# t3 as ijkabc

# JCP, 94, 442.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    _tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    ftmp = h5py.File(_tmpfile.name)
    eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), 'f8')
    orbsym = _sort_eri(mycc, eris, nocc, nvir, eris_vvop, log)

    ftmp['t2'] = t2  # read back late.  Cache t2T in t2 to reduce memory footprint
    mo_energy, t1T, t2T, vooo = _sort_t2_vooo_(mycc, orbsym, t1, t2, eris)
    cpu1 = log.timer_debug1('CCSD(T) sort_eri', *cpu1)

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
    et_sum = [0]
    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv = _ccsd.libcc.CCsd_t_contract
        drv.restype = ctypes.c_double
        et = drv(mo_energy.ctypes.data_as(ctypes.c_void_p),
                 t1T.ctypes.data_as(ctypes.c_void_p),
                 t2T.ctypes.data_as(ctypes.c_void_p),
                 vooo.ctypes.data_as(ctypes.c_void_p),
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
        et_sum[0] += et
        return et

    # The rest 20% memory for cache b
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mycc.max_memory - mem_now)
    bufsize = max(1, (max_memory*1e6/8-nocc**3*100)*.7/(nocc*nmo))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    for a0, a1 in reversed(list(lib.prange_tril(0, nvir, bufsize))):
        with lib.call_in_background(contract) as async_contract:
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            async_contract(a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                            cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/6):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                async_contract(a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                                cache_row_b,cache_col_b))
                cache_row_b = cache_col_b = None
            cache_row_a = cache_col_a = None

    t2[:] = ftmp['t2']
    ftmp.close()
    _tmpfile = None
    et = et_sum[0] * 2
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction = %.15g', et)
    return et

def _sort_eri(mycc, eris, nocc, nvir, vvop, log):
    cpu1 = (time.clock(), time.time())
    mol = mycc.mol
    nmo = nocc + nvir

    if mol.symmetry:
        orbsym = symm.addons.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                            eris.mo_coeff, check=False)
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32) % 10
    else:
        orbsym = numpy.zeros(nmo, dtype=numpy.int32)

    o_sorted = _irrep_argsort(orbsym[:nocc])
    v_sorted = _irrep_argsort(orbsym[nocc:])
    vrank = numpy.argsort(v_sorted)

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.9)
    blksize = min(nvir, max(16, int(max_memory*1e6/8/(nvir*nocc*nmo))))
    with lib.call_in_background(vvop.__setitem__) as save:
        bufopv = numpy.empty((nocc,nmo,nvir))
        buf1 = numpy.empty_like(bufopv)
        buf = numpy.empty((nocc,nvir,nvir))
        for j0, j1 in lib.prange(0, nvir, blksize):
            ovov = numpy.asarray(eris.ovov[:,j0:j1])
            ovvv = numpy.asarray(eris.ovvv[:,j0:j1])
            for j in range(j0,j1):
                oov = ovov[o_sorted,j-j0]
                ovv = lib.unpack_tril(ovvv[o_sorted,j-j0], out=buf)
                bufopv[:,:nocc,:] = oov[:,o_sorted][:,:,v_sorted]
                bufopv[:,nocc:,:] = ovv[:,v_sorted][:,:,v_sorted]
                save(vrank[j], bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    return orbsym

def _sort_t2_vooo_(mycc, orbsym, t1, t2, eris):
    ovoo = numpy.asarray(eris.ovoo)
    nocc, nvir = t1.shape
    if mycc.mol.symmetry:
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32)
        o_sorted = _irrep_argsort(orbsym[:nocc])
        v_sorted = _irrep_argsort(orbsym[nocc:])
        mo_energy = eris.fock.diagonal()
        mo_energy = numpy.hstack((mo_energy[:nocc][o_sorted],
                                  mo_energy[nocc:][v_sorted]))
        t1T = numpy.asarray(t1.T[v_sorted][:,o_sorted], order='C')

        o_sym = orbsym[o_sorted]
        oo_sym = (o_sym[:,None] ^ o_sym).ravel()
        oo_sorted = _irrep_argsort(oo_sym)
        #:vooo = eris.ovoo.transpose(1,0,2,3)
        #:vooo = vooo[v_sorted][:,o_sorted][:,:,o_sorted][:,:,:,o_sorted]
        #:vooo = vooo.reshape(nvir,-1,nocc)[:,oo_sorted]
        oo_idx = numpy.arange(nocc**2).reshape(nocc,nocc)[o_sorted][:,o_sorted]
        oo_idx = oo_idx.ravel()[oo_sorted]
        oo_idx = (oo_idx[:,None]*nocc+o_sorted).ravel()
        vooo = lib.take_2d(ovoo.transpose(1,0,2,3).reshape(nvir,-1), v_sorted, oo_idx)

        #:t2T = t2.transpose(2,3,1,0)
        #:t2T = ref_t2T[v_sorted][:,v_sorted][:,:,o_sorted][:,:,:,o_sorted]
        #:t2T = ref_t2T.reshape(nvir,nvir,-1)[:,:,oo_sorted]
        t2T = lib.transpose(t2.reshape(nocc**2,-1))
        oo_idx = numpy.arange(nocc**2).reshape(nocc,nocc).T[o_sorted][:,o_sorted]
        oo_idx = oo_idx.ravel()[oo_sorted]
        vv_idx = (v_sorted[:,None]*nvir+v_sorted).ravel()
        t2T = lib.take_2d(t2T.reshape(nvir**2,-1), vv_idx, oo_idx, out=t2)
        t2T = t2T.reshape(nvir,nvir,nocc,nocc)
    else:
        t1T = t1.T.copy()
        t2T = lib.transpose(t2.reshape(nocc**2,-1))
        t2T = lib.transpose(t2T.reshape(-1,nocc,nocc), axes=(0,2,1), out=t2)
        vooo = ovoo.transpose(1,0,2,3).copy()
        mo_energy = numpy.asarray(eris.fock.diagonal(), order='C')
    vooo = vooo.reshape(nvir,nocc,nocc,nocc)
    t2T = t2T.reshape(nvir,nvir,nocc,nocc)
    return mo_energy, t1T, t2T, vooo

def _irrep_argsort(orbsym):
    return numpy.hstack([numpy.where(orbsym == i)[0] for i in range(8)])


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
    mcc.mo_energy = mcc._scf.mo_energy = numpy.arange(0., nocc+nvir)
    eris.fock = numpy.diag(mcc.mo_energy)
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
