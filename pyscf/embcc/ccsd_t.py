import logging
import numpy as np
from pyscf import lib


from .util import einsum

__all__ = [
        "ccsd_t",
        "kernel",
        "kernel_new",
        ]

log = logging.getLogger(__name__)

def ccsd_t(t1, t2, t1loc, t2loc, eris, variant=1):
    """JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
     t3 as ijkabc
    """

    t1T = t1.T
    t1locT = t1loc.T
    t2T = t2.transpose(2,3,0,1)
    t2locT = t2loc.transpose(2,3,0,1)
    #t2locT = t2T.copy()

    nocc, nvir = t1.shape
    #mo_e = eris.fock.diagonal()
    # Correct for PBC with exxdiv??
    mo_e = eris.mo_energy
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)
    eijk2 = np.add.outer(e_occ, np.add.outer(e_occ, e_occ))
    assert np.allclose(eijk, eijk2)

    eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
    eris_vooo = np.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    eris_vvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]

    def get_w_full(a, b, c):
        w = einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w

    def get_w_loc(a, b, c):
        w = einsum('if,fkj->ijk', eris_vvov[a,b], t2locT[c,:])
        w-= einsum('ijm,mk->ijk', eris_vooo[a,:], t2locT[b,c])
        return w

    def get_v_full(a, b, c):
        v = einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v

    def get_v_loc(a, b, c):
        v = einsum('ij,k->ijk', eris_vvoo[a,b], t1locT[c])
        v+= einsum('ij,k->ijk', t2locT[a,b], fvo[c])
        return v

    if variant == 1:

        get_w = get_w_loc

        def get_z(a, b, c):
            w = get_w_full(a, b, c)
            v = get_v_full(a, b, c)
            z = r3(w + v/2)
            return z
    else:

        get_w = get_w_full

        def get_z(a, b, c):
            w = get_w_loc(a, b, c)
            v = get_v_loc(a, b, c)
            z = r3(w + v/2)
            return z

    et = 0
    for a in range(nvir):
        for b in range(a+1):
            for c in range(b+1):
                d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                if a == c:  # a == b == c
                    d3 *= 6
                elif a == b or b == c:
                    d3 *= 2

                # local W
                wabc = get_w(a, b, c)
                wacb = get_w(a, c, b)
                wbac = get_w(b, a, c)
                wbca = get_w(b, c, a)
                wcab = get_w(c, a, b)
                wcba = get_w(c, b, a)
                # Z
                zabc = get_z(a, b, c) / d3
                zacb = get_z(a, c, b) / d3
                zbac = get_z(b, a, c) / d3
                zbca = get_z(b, c, a) / d3
                zcab = get_z(c, a, b) / d3
                zcba = get_z(c, b, a) / d3

                et += einsum('ijk,ijk', wabc, zabc.conj())
                et += einsum('ikj,ijk', wacb, zabc.conj())
                et += einsum('jik,ijk', wbac, zabc.conj())
                et += einsum('jki,ijk', wbca, zabc.conj())
                et += einsum('kij,ijk', wcab, zabc.conj())
                et += einsum('kji,ijk', wcba, zabc.conj())

                if True:
                    et += einsum('ijk,ijk', wacb, zacb.conj())
                    et += einsum('ikj,ijk', wabc, zacb.conj())
                    et += einsum('jik,ijk', wcab, zacb.conj())
                    et += einsum('jki,ijk', wcba, zacb.conj())
                    et += einsum('kij,ijk', wbac, zacb.conj())
                    et += einsum('kji,ijk', wbca, zacb.conj())

                if True:
                    et += einsum('ijk,ijk', wbac, zbac.conj())
                    et += einsum('ikj,ijk', wbca, zbac.conj())
                    et += einsum('jik,ijk', wabc, zbac.conj())
                    et += einsum('jki,ijk', wacb, zbac.conj())
                    et += einsum('kij,ijk', wcba, zbac.conj())
                    et += einsum('kji,ijk', wcab, zbac.conj())

                if True:
                    et += einsum('ijk,ijk', wbca, zbca.conj())
                    et += einsum('ikj,ijk', wbac, zbca.conj())
                    et += einsum('jik,ijk', wcba, zbca.conj())
                    et += einsum('jki,ijk', wcab, zbca.conj())
                    et += einsum('kij,ijk', wabc, zbca.conj())
                    et += einsum('kji,ijk', wacb, zbca.conj())

                if True:
                    et += einsum('ijk,ijk', wcab, zcab.conj())
                    et += einsum('ikj,ijk', wcba, zcab.conj())
                    et += einsum('jik,ijk', wacb, zcab.conj())
                    et += einsum('jki,ijk', wabc, zcab.conj())
                    et += einsum('kij,ijk', wbca, zcab.conj())
                    et += einsum('kji,ijk', wbac, zcab.conj())

                    et += einsum('ijk,ijk', wcba, zcba.conj())
                    et += einsum('ikj,ijk', wcab, zcba.conj())
                    et += einsum('jik,ijk', wbca, zcba.conj())
                    et += einsum('jki,ijk', wbac, zcba.conj())
                    et += einsum('kij,ijk', wacb, zcba.conj())
                    et += einsum('kji,ijk', wabc, zcba.conj())
    et *= 2
    #log.debug('CCSD(T) correction = %.12g', et)

    return et

def r3(w):
    return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
            - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
            - 2 * w.transpose(1,0,2))


import ctypes
from pyscf.cc import _ccsd

def kernel_new(t1, t2, t2loc, eris):

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    mo_e = eris.mo_energy.copy()

    gvvov = eris.get_ovvv().conj().transpose(1,3,0,2).copy()
    gvooo = np.asarray(eris.ovoo).conj().transpose(1,0,3,2).copy()
    gvvoo = np.asarray(eris.ovov).conj().transpose(1,3,0,2).copy()

    t1T = t1.T.copy()
    t2T = t2.transpose(2,3,0,1).copy()
    #t2T = t2.transpose(2,3,0,1).copy()
    t2locT = t2loc.transpose(2,3,0,1).copy()

    fvo = eris.fock[nocc:,:nocc].copy()

    et = np.zeros(1)

    drv = _ccsd.libcc.ccsd_t_simple_emb
    drv(
        ctypes.c_int(nocc), ctypes.c_int(nvir),
        mo_e.ctypes.data_as(ctypes.c_void_p),
        #
        t1T.ctypes.data_as(ctypes.c_void_p),
        t2T.ctypes.data_as(ctypes.c_void_p),
        t2locT.ctypes.data_as(ctypes.c_void_p),
        #
        fvo.ctypes.data_as(ctypes.c_void_p),
        gvvov.ctypes.data_as(ctypes.c_void_p),
        gvooo.ctypes.data_as(ctypes.c_void_p),
        gvvoo.ctypes.data_as(ctypes.c_void_p),
        # OUT
        et.ctypes.data_as(ctypes.c_void_p),
    )

    return et[0]



if True:
    # FAST ADAPTION
    import time
    #import ctypes
    import numpy
    from pyscf import lib
    from pyscf import symm
    from pyscf.lib import logger
    #from pyscf.cc import _ccsd
    
    from pyscf.cc.ccsd_t import _sort_eri
    
    # JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
    def kernel(mycc, t1, t2, t2loc, eris, verbose=logger.NOTE):
        cpu1 = cpu0 = (time.clock(), time.time())
        log = logger.new_logger(mycc, verbose)
    
        t1 = t1.copy()
        t2 = t2.copy()
        t2loc = t2loc.copy()
    
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
    
        mo_energy, t1T, t2T, t2locT, vooo, fvo = \
                _sort_t2_vooo_(t1, t2, t2loc, eris)
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
        if dtype == numpy.complex:
            raise RuntimeError()
        else:
            drv = _ccsd.libcc.CCsd_t_contract_emb
        et_sum = numpy.zeros(1, dtype=dtype)
        def contract(a0, a1, b0, b1, cache):
            cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
            drv(et_sum.ctypes.data_as(ctypes.c_void_p),
                mo_energy.ctypes.data_as(ctypes.c_void_p),
                t1T.ctypes.data_as(ctypes.c_void_p),
                t2T.ctypes.data_as(ctypes.c_void_p),
                # NEW:
                t2locT.ctypes.data_as(ctypes.c_void_p),
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
    
        et_sum *= 2
        if abs(et_sum[0].imag) > 1e-4:
            logger.warn(mycc, 'Non-zero imaginary part of CCSD(T) energy was found %s',
                        et_sum[0])
        et = et_sum[0].real
        log.timer('CCSD(T)', *cpu0)
        log.note('CCSD(T) correction = %.15g', et)
        return et
    
    def _sort_t2_vooo_(t1, t2, t2loc, eris):
        assert(t2.flags.c_contiguous)
        assert(t2loc.flags.c_contiguous)
        vooo = numpy.asarray(eris.ovoo).transpose(1,0,3,2).conj().copy()
        nocc, nvir = t1.shape
    
        fvo = eris.fock[nocc:,:nocc].copy()
        t1T = t1.T.copy()
        t2T = lib.transpose(t2.reshape(nocc**2,nvir**2))
        t2T = lib.transpose(t2T.reshape(nvir**2,nocc,nocc), axes=(0,2,1), out=t2)
    
        t2locT = lib.transpose(t2loc.reshape(nocc**2,nvir**2))
        t2locT = lib.transpose(t2locT.reshape(nvir**2,nocc,nocc), axes=(0,2,1), out=t2loc)
    
        mo_energy = numpy.asarray(eris.mo_energy, order='C')
        t2T = t2T.reshape(nvir,nvir,nocc,nocc)
        t2locT = t2locT.reshape(nvir,nvir,nocc,nocc)
    
        return mo_energy, t1T, t2T, t2locT, vooo, fvo
