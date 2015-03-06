#!/usr/bin/env python

import os, sys
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
    aapp, appa, Iapcv = _trans_aapp_(mo, ncore, ncas, load_buf)
    vhf_c, j_cp, k_cp, Icvcv = _trans_cvcv_(mo, ncore, ncas, load_buf)
    return vhf_c, j_cp, k_cp, aapp, appa, Iapcv, Icvcv


def trans_e1_outcore(mol, mo, ncore, ncas,
                     max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=logger.WARN):
    time0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo), swapfile.name,
                                max_memory=max_memory, ioblk_size=ioblk_size,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(bfn_id):
        if log.verbose >= logger.DEBUG1:
            time1[:] = log.timer('between load_buf', *tuple(time1))
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat[bfn_id*nmo:(bfn_id+1)*nmo]
            col0 = col1
        if log.verbose >= logger.DEBUG1:
            time1[:] = log.timer('load_buf', *tuple(time1))
        return buf
    time0 = log.timer('halfe1', *time0)
    time1 = [time.clock(), time.time()]
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    aapp, appa, Iapcv = _trans_aapp_(mo, ncore, ncas, load_buf, ao_loc)
    time0 = log.timer('trans_aapp', *time0)
    vhf_c, j_cp, k_cp, Icvcv = _trans_cvcv_(mo, ncore, ncas, load_buf, ao_loc)
    time0 = log.timer('trans_cvcv', *time0)
    fswap.close()
    return vhf_c, j_cp, k_cp, aapp, appa, Iapcv, Icvcv


# approx = 0: all, includes Icvcv etc
# approx = 1: aapp, appa and vhf, jcp, kcp
# approx = 2: vhf, aapp, appa
def light_e1_outcore(mol, mo, ncore, ncas,
                     max_memory=None, approx=1, verbose=logger.WARN):
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

    mo = numpy.asarray(mo, order='F')
    nao, nmo = mo.shape
    nij_pair = ncas * nmo
    pashape = (0, nmo, ncore, ncas)
    if approx == 1:
        jc = numpy.empty((nao,nao,ncore))
        kc = numpy.zeros((nao,nao,ncore))
    else:
        jc = numpy.empty((nao,nao))
        kc = numpy.zeros((nao,nao))

    mem_words = int(max_memory * 1e6/8)
    aobuflen = mem_words//(nao*nao+nocc*nmo) + 1
    shranges = outcore.guess_shell_ranges(mol, aobuflen, aobuflen, 's4')
    ao2mopt = _ao2mo.AO2MOpt(mol, 'cint2e_sph',
                             'CVHFnr_schwarz_cond', 'CVHFsetnr_direct_scf')
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    log.debug('mem cache %.8g MB', mem_words*8/1e6)
    ti0 = log.timer('Initializing light_e1_outcore', *time0)
    klaoblks = nstep = len(shranges)
    paapp = 0
    for istep,sh_range in enumerate(shranges):
        log.debug('[%d/%d], AO [%d:%d], len(buf) = %d',
                  istep+1, nstep, *(sh_range[:3]))
        buf = numpy.empty((sh_range[2],nao*nao))
        _ao2mo.nr_e1fill_('cint2e_sph', sh_range[:3],
                          mol._atm, mol._bas, mol._env, 's4', 1, ao2mopt, buf)
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('AO integrals buffer', *ti0)
        bufpa = _ao2mo.nr_e1_(buf, mo, pashape, 's4', 's1')
# aapp, appa
        aapp_buf[paapp:paapp+sh_range[2]] = \
                bufpa.reshape(sh_range[2],nmo,ncas)[:,ncore:nocc]
        paapp += sh_range[2]
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('aapp buffer', *ti1)

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
                mo1 = mo[j0:j1,ncore:nocc]
                for i in range(di):
                    appa_buf[:,i0+i] += numpy.dot(mo1.T, buf1[i])
                buf1 = bufpa[p0:p0+dij].reshape(di,-1)
            appa_buf[:,j0:j1] += \
                    numpy.dot(mo[i0:i1,ncore:nocc].T, buf1).reshape(ncas,dj,-1)
            p0 += dij
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('appa buffer', *ti1)

# jc_pp, kc_pp
        buf1 = numpy.empty((sh_range[2],nao*ncore))
        fmmm = _fpointer('MCSCFhalfmmm_nr_s2_ket')
        ftrans = _fpointer('AO2MOtranse1_nr_s4')
        fdrv = getattr(libmcscf, 'AO2MOnr_e2_drv')
        fdrv(ftrans, fmmm,
             buf1.ctypes.data_as(ctypes.c_void_p),
             buf.ctypes.data_as(ctypes.c_void_p),
             mo.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(sh_range[2]), ctypes.c_int(nao),
             ctypes.c_int(0), ctypes.c_int(nao),
             ctypes.c_int(0), ctypes.c_int(ncore),
             ctypes.c_void_p(0), ctypes.c_int(0))
        buf = buf1
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('halfmmm core (uv|tc)', *ti1)
        if approx == 1:
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
                    buf1 = numpy.empty((di,di,nao*ncore))
                    idx = numpy.tril_indices(di)
                    buf1[idx] = buf[p0:p0+dij]
                    buf1[idx[1],idx[0]] = buf[p0:p0+dij]
                    buf1 = buf1.reshape(di,di,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uvc', buf1, mo[:,:ncore])
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vpc', buf1, mo[i0:i1,:ncore])
                else:
                    dij = di * dj
                    buf1 = buf[p0:p0+dij].reshape(di,dj,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uvc', buf1, mo[:,:ncore])
                    jc[j0:j1,i0:i1] = jc[i0:i1,j0:j1].transpose(1,0,2)
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vpc', buf1, mo[i0:i1,:ncore])
                    kc[i0:i1] += numpy.einsum('uvpc,vc->upc', buf1, mo[j0:j1,:ncore])
                p0 += dij
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('jc and kc buffer', *ti1)
        else:
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
                    buf1 = numpy.empty((di,di,nao*ncore))
                    idx = numpy.tril_indices(di)
                    buf1[idx] = buf[p0:p0+dij]
                    buf1[idx[1],idx[0]] = buf[p0:p0+dij]
                    buf1 = buf1.reshape(di,di,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uv', buf1, mo[:,:ncore])
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vp', buf1, mo[i0:i1,:ncore])
                else:
                    dij = di * dj
                    buf1 = buf[p0:p0+dij].reshape(di,dj,nao,ncore)
                    jc[i0:i1,j0:j1] = numpy.einsum('uvpc,pc->uv', buf1, mo[:,:ncore])
                    kc[j0:j1] += numpy.einsum('uvpc,uc->vp', buf1, mo[i0:i1,:ncore])
                    kc[i0:i1] += numpy.einsum('uvpc,vc->up', buf1, mo[j0:j1,:ncore])
                p0 += dij
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('jc and kc buffer', *ti1)

        buf1 = bufpa = None
        ti0 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)

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

    if approx == 1:
        vhf_c = numpy.einsum('ijc->ij', jc)*2 - numpy.einsum('ijc->ij', kc)
        vhf_c = reduce(numpy.dot, (mo.T, vhf_c, mo))
        j_cp = numpy.dot(mo.T, jc.reshape(nao,-1)).reshape(nao,nao,ncore)
        j_cp = numpy.einsum('pj,jpi->ij', mo, j_cp)
        k_cp = numpy.dot(mo.T, kc.reshape(nao,-1)).reshape(nao,nao,ncore)
        k_cp = numpy.einsum('pj,jpi->ij', mo, k_cp)
    else:
        jc = pyscf.lib.hermi_triu(jc, hermi=1, inplace=True)
        vhf_c = reduce(numpy.dot, (mo.T, jc*2-kc, mo))
        j_cp = k_cp = None

    time0 = log.timer('mc_ao2mo', *time0)
    return vhf_c, j_cp, k_cp, aapp, appa


def _trans_aapp_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    klshape = (0, nmo, 0, nmo)

    japcv = numpy.empty((ncas,nmo,ncore,nmo-ncore))
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
        #japcv = apcv * 4 - acpv.transpose(0,2,1,3) - avcp.transpose(0,3,2,1)
        japcv[i] = ppp[:,:ncore,ncore:] * 4 \
                 - ppp[:ncore,:,ncore:].transpose(1,0,2) \
                 - ppp[ncore:,:ncore,:].transpose(2,1,0)
    return aapp, appa, japcv

def _trans_cvcv_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    jcvcv = numpy.zeros((ncore,nmo-ncore,ncore,nmo-ncore))
    vcp = numpy.empty((nmo-ncore,ncore,nmo))
    cpp = numpy.empty((ncore,nmo,nmo))
    vj = numpy.zeros((nmo,nmo))
    vk = numpy.zeros((nmo,nmo))
    j_cp = numpy.zeros((ncore,nmo))
    k_cp = numpy.zeros((ncore,nmo))
    for i in range(ncore):
        buf = fload(i)
        klshape = (0, ncore, 0, nmo)
        _ao2mo.nr_e2_(buf[ncore:nmo], mo, klshape,
                      aosym='s4', mosym='s1', vout=vcp, ao_loc=ao_loc)
        vk[ncore:] += vcp[:,i]
        k_cp[i,ncore:] = vcp[:,i,ncore:].diagonal()

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2_(buf[:ncore], mo, klshape,
                      aosym='s4', mosym='s2', vout=buf[:ncore], ao_loc=ao_loc)
        for j in range(ncore):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    cpp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        vj += cpp[i]
        j_cp[i] = cpp[i].diagonal()
        vk[:ncore] += cpp[:,i]
        k_cp[i,:ncore] = cpp[:,i,:ncore].diagonal()

        #jcvcv = cvcv * 4 - cvcv.transpose(0,3,2,1) - ccvv.transpose(0,2,1,3)
        jcvcv[i] = vcp[:,:,ncore:] * 4 \
                 - vcp[:,:,ncore:].transpose(2,1,0) \
                 - cpp[:,ncore:,ncore:].transpose(1,0,2)
    return vj*2-vk, j_cp, k_cp, jcvcv



# approx = 0: all, includes Icvcv etc
# approx = 1: aapp, appa and vhf, jcp, kcp
# approx = 2: vhf, aapp, appa
class _ERIS(object):
    def __init__(self, casscf, mo, method='incore', approx=0):
        mol = casscf.mol
        self.ncore = casscf.ncore
        self.ncas = casscf.ncas
        nmo = mo.shape[1]
        ncore = self.ncore
        ncas = self.ncas

        if method == 'outcore' \
           or _mem_usage(ncore, ncas, nmo)[0] + nmo**4*2/1e6 > casscf.max_memory*.9 \
           or casscf._scf._eri is None:
            log = logger.Logger(casscf.stdout, casscf.verbose)
            if approx == 0:
                self.vhf_c, self.j_cp, self.k_cp, self.aapp, self.appa, \
                self.Iapcv, self.Icvcv = \
                        trans_e1_outcore(casscf.mol, mo, casscf.ncore, casscf.ncas,
                                         max_memory=casscf.max_memory, verbose=log)
            else:
                self.vhf_c, self.j_cp, self.k_cp, self.aapp, self.appa = \
                        light_e1_outcore(casscf.mol, mo, casscf.ncore, casscf.ncas,
                                         max_memory=casscf.max_memory,
                                         approx=approx, verbose=log)
        elif method == 'incore' and casscf._scf._eri is not None:
            self.vhf_c, self.j_cp, self.k_cp, self.aapp, self.appa, \
            self.Iapcv, self.Icvcv = \
                    trans_e1_incore(casscf._scf._eri, mo,
                                    casscf.ncore, casscf.ncas)
        else:
            raise KeyError('update ao2mo')

def _mem_usage(ncore, ncas, nmo):
    nvir = nmo - ncore
    outcore = (ncore**2*nvir**2 + ncas*nmo*ncore*nvir + ncore*nmo**2*3 +
               ncas**2*nmo**2*2 + nmo**3*2) * 8/1e6
    incore = outcore + nmo**4/1e6 + ncore*nmo**3*4/1e6
    if outcore > 10000:
        sys.stderr.write('Be careful with the virtual memorty address space `ulimit -v`\n')
    return incore, outcore


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
    eris2 = _ERIS(mc, mo, 'outcore', approx=1)
    eris3 = _ERIS(mc, mo, 'outcore', approx=2)
    print('vhf_c', numpy.allclose(eris0.vhf_c, eris1.vhf_c))
    print('j_cp ', numpy.allclose(eris0.j_cp , eris1.j_cp ))
    print('k_cp ', numpy.allclose(eris0.k_cp , eris1.k_cp ))
    print('aapp ', numpy.allclose(eris0.aapp , eris1.aapp ))
    print('appa ', numpy.allclose(eris0.appa , eris1.appa ))
    print('Iapcv', numpy.allclose(eris0.Iapcv, eris1.Iapcv))
    print('Icvcv', numpy.allclose(eris0.Icvcv, eris1.Icvcv))

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

    capv = eri[:ncore,ncore:nocc,:,ncore:]
    cvap = eri[:ncore,ncore:,ncore:nocc,:]
    cpav = eri[:ncore,:,ncore:nocc,ncore:]
    ccvv = eri[:ncore,:ncore,ncore:,ncore:]
    cvcv = eri[:ncore,ncore:,:ncore,ncore:]

    cVAp = cvap * 4 \
         - capv.transpose(0,3,1,2) \
         - cpav.transpose(0,3,2,1)
    cVCv = cvcv * 4 \
         - ccvv.transpose(0,3,1,2) \
         - cvcv.transpose(0,3,2,1)

    print('vhf_c', numpy.allclose(vhf_c, eris1.vhf_c))
    print('j_cp ', numpy.allclose(j_cp, eris1.j_cp))
    print('k_cp ', numpy.allclose(k_cp, eris1.k_cp))
    print('aapp ', numpy.allclose(aapp , eris0.aapp ))
    print('appa ', numpy.allclose(appa , eris0.appa ))
    print('Iapcv', numpy.allclose(cVAp.transpose(2,3,0,1), eris1.Iapcv ))
    print('Icvcv', numpy.allclose(cVCv.transpose(2,3,0,1), eris1.Icvcv ))

