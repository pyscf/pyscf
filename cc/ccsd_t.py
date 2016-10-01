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
from pyscf.cc import _ccsd
from pyscf.ao2mo.outcore import balance_partition

'''
CCSD(T)
'''

# t3 as ijkabc

# JCP, 94, 442.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    _tmpfile = tempfile.NamedTemporaryFile()
    ftmp = h5py.File(_tmpfile.name)
    ftmp['t2'] = t2  # read back late.  Cache t2T in t2 to reduce memory footprint
    eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), 'f8')

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, int(max_memory*.6e9/8/(nvir*nocc*nmo)))
    buf = numpy.empty((blksize,nvir,nocc,nmo))
    ovvbuf = numpy.empty((nvir,nvir,nvir))
    for j0, j1 in lib.prange(0, nvir, blksize):
        vvopbuf = numpy.ndarray((j1-j0,nvir,nocc,nmo), buffer=buf)
        for i in range(nocc):
            tmp = numpy.asarray(eris.ovov[i*nvir+j0:i*nvir+j1])
            vvopbuf[:,:,i,:nocc] = tmp.reshape(j1-j0,nocc,nvir).transpose(0,2,1)
            tmp = lib.unpack_tril(eris.ovvv[i*nvir+j0:i*nvir+j1],out=ovvbuf)
            vvopbuf[:,:,i,nocc:] = tmp.reshape(j1-j0,nvir,nvir)
        eris_vvop[j0:j1] = vvopbuf
    vvopbuf = tmp = ovvbuf = buf = None
    eris_vooo = numpy.asarray(eris.ovoo.transpose(1,0,2,3), order='C')

    t1T = t1.T.copy()
    t2T = t2.transpose(1,0,2,3).copy().reshape(nocc**2,-1)
    t2T = lib.transpose(t2T, out=t2).reshape(nvir,nvir,nocc,nocc)

    # The rest 20% memory for cache b
    bufsize = (max_memory*1e6/8/(nocc*nmo) - nocc**3*100) * .8
    def tril_prange(start, stop, step):
        cum_costs = numpy.arange(stop+1)**2
        tasks = balance_partition(cum_costs, step, start, stop)
        return tasks

    def contract(a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv = _ccsd.libcc.CCsd_t_contract
        drv.restype = ctypes.c_double
        et = drv(mycc._scf.mo_energy.ctypes.data_as(ctypes.c_void_p),
                 t1T.ctypes.data_as(ctypes.c_void_p),
                 t2T.ctypes.data_as(ctypes.c_void_p),
                 eris_vooo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nocc), ctypes.c_int(nvir),
                 ctypes.c_int(a0), ctypes.c_int(a1),
                 ctypes.c_int(b0), ctypes.c_int(b1),
                 cache_row_a.ctypes.data_as(ctypes.c_void_p),
                 cache_col_a.ctypes.data_as(ctypes.c_void_p),
                 cache_row_b.ctypes.data_as(ctypes.c_void_p),
                 cache_col_b.ctypes.data_as(ctypes.c_void_p))
        return et

    et = 0
    handler = None
    cache_col_a = cache_col_b = numpy.zeros((0,))
    for a0, a1, na in reversed(tril_prange(0, nvir, bufsize)):
        if handler is not None:
            et += handler.get()
        # DO NOT prefetch here to reserve more memory to cache a
        cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1])
        if a0 > 0:
            cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1])
        handler = lib.background_thread(contract, a0, a1, a0, a1,
                                        (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

        for b0, b1, nb in tril_prange(0, a0, bufsize/10):
            cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1])
            if b0 > 0:
                cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1])
            if handler is not None:
                et += handler.get()
            handler = lib.background_thread(contract, a0, a1, b0, b1,
                                            (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
            cache_row_b = cache_col_b = None
        cache_row_a = cache_col_a = None
    if handler is not None:
        et += handler.get()

    t2[:] = ftmp['t2']
    ftmp.close()
    _tmpfile = None
    return et*2

def permute_contract(z, w):
    z0, z1, z2, z3, z4, z5 = z
    et = numpy.einsum('ijk,ijk', z[0], w)
    et+= numpy.einsum('ikj,ijk', z[1], w)
    et+= numpy.einsum('jik,ijk', z[2], w)
    et+= numpy.einsum('jki,ijk', z[3], w)
    et+= numpy.einsum('kij,ijk', z[4], w)
    et+= numpy.einsum('kji,ijk', z[5], w)
    return et


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.M()
    numpy.random.seed(12)
    nocc, nvir = 5, 12
    eris = lambda :None
    eris.ovvv = numpy.random.random((nocc*nvir,nvir*(nvir+1)//2)) * .1
    eris.ovoo = numpy.random.random((nocc,nvir,nocc,nocc)) * .1
    eris.ovov = numpy.random.random((nocc*nvir,nocc*nvir)) * .1
    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    mf = scf.RHF(mol)
    mcc = cc.CCSD(mf)
    mcc._scf.mo_energy = numpy.arange(0., nocc+nvir)
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
    print(mcc.ecc)

    e3b = ccsd_t_o0.kernel2(mcc, mcc.ao2mo())
    print(e3b, mcc.ecc+e3b)
    e3a = kernel(mcc, mcc.ao2mo())
    print(e3a, mcc.ecc+e3a)

