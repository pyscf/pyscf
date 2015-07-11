#!/usr/bin/env python

import sys
import ctypes
import _ctypes
import time
import tempfile
from functools import reduce
import numpy
import h5py
import pyscf.lib
import pyscf.lib.numpy_helper
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore

# least memory requirements:
# nmo  ncore  ncas  outcore  incore
# 200  40     16    0.8GB    3.7 GB (_eri 1.6GB intermediates 1.3G)
# 250  50     16    1.7GB    8.2 GB (_eri 3.9GB intermediates 2.6G)
# 300  60     16    3.1GB    16.8GB (_eri 8.1GB intermediates 5.6G)
# 400  80     16    8.5GB    53  GB (_eri 25.6GB intermediates 19G)
# 500  100    16    19 GB
# 600  120    16    37 GB
# 750  150    16    85 GB

libmcscf = pyscf.lib.load_library('libmcscf')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libmcscf._handle, name))

def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    eri1 = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo), compact=False)

    load_buf = lambda bufid: eri1[bufid*nmo:bufid*nmo+nmo]
    aapp, appa = _trans_aapp_(mo, ncore, ncas, load_buf)
    vhf_c, j_cp, k_cp = _trans_cvcv_(mo, ncore, ncas, load_buf)
    return vhf_c, j_cp, k_cp, aapp, appa


# level = 1: aapp, appa and vhf, jcp, kcp
# level = 2 or 3: aapp, appa
def trans_e1_outcore(mol, mo, ncore, ncas,
                     max_memory=None, level=1, verbose=logger.WARN):
    time0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas
    aapp_buf = numpy.empty((nao_pair,ncas,ncas))
    appa_buf = numpy.zeros((ncas,nao,nmo*ncas))
    max_memory -= (aapp_buf.nbytes+appa_buf.nbytes) / 1e6

    mo = numpy.asarray(mo, order='F')
    nao, nmo = mo.shape
    pashape = (0, nmo, ncore, ncas)
    if level == 1:
        jc = numpy.empty((nao,nao,ncore))
        kc = numpy.zeros((nao,nao,ncore))
        max_memory -= (jc.nbytes+kc.nbytes) / 1e6

    mem_words = int(max(1000,max_memory)*1e6/8)
    aobuflen = mem_words//(nao_pair+nocc*nmo) + 1
    shranges = outcore.guess_shell_ranges(mol, aobuflen, aobuflen, 's4')
    ao2mopt = _ao2mo.AO2MOpt(mol, 'cint2e_sph',
                             'CVHFnr_schwarz_cond', 'CVHFsetnr_direct_scf')
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    log.debug('mem cache %.8g MB', mem_words*8/1e6)
    ti0 = log.timer('Initializing trans_e1_outcore', *time0)
    nstep = len(shranges)
    paapp = 0
    maxbuflen = max([x[2] for x in shranges])
    bufs1 = numpy.empty((maxbuflen, nao_pair))
    bufs2 = numpy.empty((maxbuflen, pashape[1]*pashape[3]))
    bufs3 = numpy.empty((maxbuflen, nao*ncore))

    # fmmm, ftrans, fdrv for level 1
    fmmm = _fpointer('MCSCFhalfmmm_nr_s2_ket')
    ftrans = _fpointer('AO2MOtranse1_nr_s4')
    fdrv = getattr(libmcscf, 'AO2MOnr_e2_drv')
    for istep,sh_range in enumerate(shranges):
        log.debug('[%d/%d], AO [%d:%d], len(buf) = %d',
                  istep+1, nstep, *(sh_range[:3]))
        buf = bufs1[:sh_range[2]]
        _ao2mo.nr_e1fill_('cint2e_sph', sh_range[:3],
                          mol._atm, mol._bas, mol._env, 's4', 1, ao2mopt, buf)
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('AO integrals buffer', *ti0)
        bufpa = bufs2[:sh_range[2]]
# jc_pp, kc_pp
        if level == 1: # aapp, appa and vhf, jcp, kcp
            _ao2mo.nr_e1_(buf, mo, pashape, 's4', 's1', vout=bufpa)
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('buffer-pa', *ti1)
            buf1 = bufs3[:sh_range[2]]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 buf.ctypes.data_as(ctypes.c_void_p),
                 mo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(sh_range[2]), ctypes.c_int(nao),
                 ctypes.c_int(0), ctypes.c_int(nao),
                 ctypes.c_int(0), ctypes.c_int(ncore),
                 ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_int(0))
            p0 = 0
            for ij in range(sh_range[0], sh_range[1]):
                i,j = _ao2mo._extract_pair(ij)
                i0 = ao_loc[i]
                j0 = ao_loc[j]
                i1 = ao_loc[i+1]
                j1 = ao_loc[j+1]
                di = i1 - i0
                dj = j1 - j0
                if i == j:
                    dij = di * (di+1) // 2
                    buf = numpy.empty((di,di,nao*ncore))
                    idx = numpy.tril_indices(di)
                    buf[idx] = buf1[p0:p0+dij]
                    buf[idx[1],idx[0]] = buf1[p0:p0+dij]
                    buf = buf.reshape(di,di,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uvc', buf, mo[:,:ncore])
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vpc', buf, mo[i0:i1,:ncore])
                else:
                    dij = di * dj
                    buf = buf1[p0:p0+dij].reshape(di,dj,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uvc', buf, mo[:,:ncore])
                    jc[j0:j1,i0:i1] = jc[i0:i1,j0:j1].transpose(1,0,2)
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vpc', buf, mo[i0:i1,:ncore])
                    kc[i0:i1] += numpy.einsum('uvpc,vc->upc', buf, mo[j0:j1,:ncore])
                p0 += dij
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('jc and kc buffer', *ti1)
        else: # aapp, appa
            _ao2mo.nr_e1_(buf, mo, pashape, 's4', 's1', vout=bufpa)

# aapp, appa
        aapp_buf[paapp:paapp+sh_range[2]] = \
                bufpa.reshape(sh_range[2],nmo,ncas)[:,ncore:nocc]
        paapp += sh_range[2]
        p0 = 0
        for ij in range(sh_range[0], sh_range[1]):
            i,j = _ao2mo._extract_pair(ij)
            i0 = ao_loc[i]
            j0 = ao_loc[j]
            i1 = ao_loc[i+1]
            j1 = ao_loc[j+1]
            di = i1 - i0
            dj = j1 - j0
            if i == j:
                dij = di * (di+1) // 2
                buf1 = numpy.empty((di,di,nmo*ncas))
                idx = numpy.tril_indices(di)
                buf1[idx] = bufpa[p0:p0+dij]
                buf1[idx[1],idx[0]] = bufpa[p0:p0+dij]
                buf1 = buf1.reshape(di,-1)
            else:
                dij = di * dj
                buf1 = bufpa[p0:p0+dij].reshape(di,dj,-1)
                mo1 = mo[j0:j1,ncore:nocc].copy()
                for i in range(di):
                    appa_buf[:,i0+i] += pyscf.lib.dot(mo1.T, buf1[i])
                buf1 = bufpa[p0:p0+dij].reshape(di,-1)
            mo1 = mo[i0:i1,ncore:nocc].copy()
            appa_buf[:,j0:j1] += pyscf.lib.dot(mo1.T, buf1).reshape(ncas,dj,-1)
            p0 += dij
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('aapp and appa buffer', *ti1)

        ti0 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)
    bufs1 = bufs2 = bufs3 = None

    aapp_buf = pyscf.lib.transpose(aapp_buf.reshape(nao_pair,-1))
    aapp = _ao2mo.nr_e2_(aapp_buf, mo, (0,nmo,0,nmo), 's4', 's1', ao_loc=ao_loc)
    aapp = aapp.reshape(ncas,ncas,nmo,nmo)
    aapp_buf = None
    if nao == nmo:
        appa = appa_buf
    else:
        appa = numpy.empty((ncas,nao,nmo*ncas))
    for i in range(ncas):
        appa[i] = numpy.dot(mo.T, appa_buf[i].reshape(nao,-1))
    appa = appa.reshape(ncas,nmo,nmo,ncas)
    appa_buf = None

    if level == 1:
        vhf_c = numpy.einsum('ijc->ij', jc)*2 - numpy.einsum('ijc->ij', kc)
        vhf_c = reduce(numpy.dot, (mo.T, vhf_c, mo))
        j_cp = numpy.dot(mo.T, jc.reshape(nao,-1)).reshape(nao,nao,ncore)
        j_cp = numpy.einsum('pj,jpi->ij', mo, j_cp)
        k_cp = numpy.dot(mo.T, kc.reshape(nao,-1)).reshape(nao,nao,ncore)
        k_cp = numpy.einsum('pj,jpi->ij', mo, k_cp)
    else:
        vhf_c = j_cp = k_cp = None

    time0 = log.timer('mc_ao2mo', *time0)
    return vhf_c, j_cp, k_cp, aapp, appa


def _trans_aapp_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[1]
    nocc = ncore + ncas

    klppshape = (0, nmo, 0, nmo)
    klpashape = (0, nmo, ncore, ncas)
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    appa = numpy.empty((ncas,nmo,nmo,ncas))
    for i in range(ncas):
        apbuf = fload(ncore+i)
        _ao2mo.nr_e2_(apbuf[ncore:nocc], mo, klppshape,
                      aosym='s4', mosym='s1', ao_loc=ao_loc, vout=aapp[i])

        _ao2mo.nr_e2_(apbuf, mo, klpashape,
                      aosym='s4', mosym='s1', ao_loc=ao_loc, vout=appa[i])
    return aapp, appa

def _trans_cvcv_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[1]
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril
    klppshape = (0, nmo, 0, nmo)

    pp = numpy.empty((nmo,nmo))
    vj = numpy.zeros((nmo,nmo))
    vk = numpy.zeros((nmo,nmo))
    j_cp = numpy.zeros((ncore,nmo))
    k_cp = numpy.zeros((ncore,nmo))
    for i in range(ncore):
        buf = fload(i)
        _ao2mo.nr_e2_(buf[i:i+1], mo, klppshape,
                      aosym='s4', mosym='s1', vout=pp, ao_loc=ao_loc)
        vj += pp
        j_cp[i] = pp.diagonal()

        klshape = (i, 1, 0, nmo)
        _ao2mo.nr_e2_(buf, mo, klshape,
                      aosym='s4', mosym='s1', vout=pp, ao_loc=ao_loc)
        vk += pp
        k_cp[i] = pp.diagonal()
    return vj*2-vk, j_cp, k_cp



# level = 1: aapp, appa and vhf, jcp, kcp
# level = 2: aapp, appa, vhf
# level = 3: aapp, appa
class _ERIS(object):
    def __init__(self, casscf, mo, method='incore', level=1):
        mol = casscf.mol
        nao, nmo = mo.shape
        ncore = casscf.ncore
        ncas = casscf.ncas
        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = pyscf.lib.current_memory()[0]

        eri = casscf._scf._eri
        if (method == 'incore' and eri is not None and
            (mem_incore+mem_now < casscf.max_memory*.9) or
            mol.incore_anyway):
            if eri is None:
                from pyscf.scf import _vhf
                eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            self.vhf_c, self.j_cp, self.k_cp, self.aapp, self.appa = \
                    trans_e1_incore(eri, mo, casscf.ncore, casscf.ncas)
        else:
            import gc
            gc.collect()
            log = logger.Logger(casscf.stdout, casscf.verbose)
            max_memory = max(2000, casscf.max_memory*.9-mem_now)
            if max_memory < mem_basic:
                log.warn('Not enough memory! You need increase CASSCF.max_memory')
            self.vhf_c, self.j_cp, self.k_cp, self.aapp, self.appa = \
                    trans_e1_outcore(mol, mo, casscf.ncore, casscf.ncas,
                                     max_memory=max_memory-mem_basic,
                                     level=level, verbose=log)
            if level == 2:
                dm_core = numpy.dot(mo[:,:ncore], mo[:,:ncore].T) * 2
                vj, vk = casscf._scf.get_jk(mol, dm_core)
                self.vhf_c = reduce(numpy.dot, (mo.T, vj-vk*.5, mo))

def _mem_usage(ncore, ncas, nmo):
    nvir = nmo - ncore
    basic = ncas**2*nmo**2*2 * 8/1e6
    outcore = basic + ncore*nmo**2*3 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    if outcore > 10000:
        sys.stderr.write('Be careful with the virtual memorty address space `ulimit -v`\n')
    return incore, outcore, basic


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import ao2mo
    from pyscf.mcscf import mc1step

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()

    eris0 = _ERIS(mc, mo, 'incore')
    eris1 = _ERIS(mc, mo, 'outcore')
    eris2 = _ERIS(mc, mo, 'outcore', level=1)
    eris3 = _ERIS(mc, mo, 'outcore', level=2)
    print('vhf_c', numpy.allclose(eris0.vhf_c, eris1.vhf_c))
    print('j_cp ', numpy.allclose(eris0.j_cp , eris1.j_cp ))
    print('k_cp ', numpy.allclose(eris0.k_cp , eris1.k_cp ))
    print('aapp ', numpy.allclose(eris0.aapp , eris1.aapp ))
    print('appa ', numpy.allclose(eris0.appa , eris1.appa ))

    print('vhf_c', numpy.allclose(eris0.vhf_c, eris2.vhf_c))
    print('j_cp ', numpy.allclose(eris0.j_cp , eris2.j_cp ))
    print('k_cp ', numpy.allclose(eris0.k_cp , eris2.k_cp ))
    print('aapp ', numpy.allclose(eris0.aapp , eris2.aapp ))
    print('appa ', numpy.allclose(eris0.appa , eris2.appa ))

    print('vhf_c', numpy.allclose(eris0.vhf_c, eris3.vhf_c))
    print('aapp ', numpy.allclose(eris0.aapp , eris3.aapp ))
    print('appa ', numpy.allclose(eris0.appa , eris3.appa ))

    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    eri = ao2mo.incore.full(m._eri, mo, compact=False).reshape((nmo,)*4)
    aaap = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
    aapp = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
    appa = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
    jc_pp = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
    kc_pp = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
    vhf_c = numpy.einsum('cij->ij', jc_pp)*2 - numpy.einsum('cij->ij', kc_pp)
    j_cp = numpy.einsum('ijj->ij', jc_pp)
    k_cp = numpy.einsum('ijj->ij', kc_pp)

    print('vhf_c', numpy.allclose(vhf_c, eris1.vhf_c))
    print('j_cp ', numpy.allclose(j_cp, eris1.j_cp))
    print('k_cp ', numpy.allclose(k_cp, eris1.k_cp))
    print('aapp ', numpy.allclose(aapp , eris0.aapp ))
    print('appa ', numpy.allclose(appa , eris0.appa ))

