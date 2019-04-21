#!/usr/bin/env python
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

import sys
import ctypes
import time
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
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

libmcscf = lib.load_library('libmcscf')

def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    eri1 = ao2mo.incore.half_e1(eri_ao, (mo,mo[:,:nocc]), compact=False)
    eri1 = eri1.reshape(nmo,nocc,-1)

    klppshape = (0, nmo, 0, nmo)
    klpashape = (0, nmo, ncore, nocc)
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    for i in range(ncas):
        _ao2mo.nr_e2(eri1[ncore+i,ncore:nocc], mo, klppshape,
                      aosym='s4', mosym='s1', out=aapp[i])
    ppaa = lib.transpose(aapp.reshape(ncas*ncas,-1)).reshape(nmo,nmo,ncas,ncas)
    aapp = None

    papa = numpy.empty((nmo,ncas,nmo,ncas))
    for i in range(nmo):
        _ao2mo.nr_e2(eri1[i,ncore:nocc], mo, klpashape,
                      aosym='s4', mosym='s1', out=papa[i])

    pp = numpy.empty((nmo,nmo))
    j_cp = numpy.zeros((ncore,nmo))
    k_pc = numpy.zeros((nmo,ncore))
    for i in range(ncore):
        _ao2mo.nr_e2(eri1[i,i:i+1], mo, klppshape, aosym='s4', mosym='s1', out=pp)
        j_cp[i] = pp.diagonal()
    j_pc = j_cp.T.copy()

    pp = numpy.empty((ncore,ncore))
    for i in range(nmo):
        klshape = (i, i+1, 0, ncore)
        _ao2mo.nr_e2(eri1[i,:ncore], mo, klshape, aosym='s4', mosym='s1', out=pp)
        k_pc[i] = pp.diagonal()
    return j_pc, k_pc, ppaa, papa


# level = 1: ppaa, papa and jpc, kpc
# level > 1: ppaa, papa only.  It affects accuracy of hdiag
def trans_e1_outcore(mol, mo, ncore, ncas, erifile,
                     max_memory=None, level=1, verbose=logger.WARN):
    time0 = (time.clock(), time.time())
    log = logger.new_logger(mol, verbose)
    log.debug1('trans_e1_outcore level %d  max_memory %d', level, max_memory)
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas

    faapp_buf = lib.H5TmpFile()
    if isinstance(erifile, h5py.Group):
        feri = erifile
    else:
        feri = lib.H5TmpFile(erifile, 'w')

    mo_c = numpy.asarray(mo, order='C')
    mo = numpy.asarray(mo, order='F')
    pashape = (0, nmo, ncore, nocc)
    papa_buf = numpy.zeros((nao,ncas,nmo*ncas))
    j_pc = numpy.zeros((nmo,ncore))
    k_pc = numpy.zeros((nmo,ncore))

    mem_words = int(max(2000,max_memory-papa_buf.nbytes/1e6)*1e6/8)
    aobuflen = mem_words//(nao_pair+nocc*nmo) + 1
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    shranges = outcore.guess_shell_ranges(mol, True, aobuflen, None, ao_loc)
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor,
                             'CVHFnr_schwarz_cond', 'CVHFsetnr_direct_scf')
    nstep = len(shranges)
    paapp = 0
    maxbuflen = max([x[2] for x in shranges])
    log.debug('mem_words %.8g MB, maxbuflen = %d', mem_words*8/1e6, maxbuflen)
    bufs1 = numpy.empty((maxbuflen, nao_pair))
    bufs2 = numpy.empty((maxbuflen, nmo*ncas))
    if level == 1:
        bufs3 = numpy.empty((maxbuflen, nao*ncore))
        log.debug('mem cache %.8g MB',
                  (bufs1.nbytes+bufs2.nbytes+bufs3.nbytes)/1e6)
    else:
        log.debug('mem cache %.8g MB', (bufs1.nbytes+bufs2.nbytes)/1e6)
    ti0 = log.timer('Initializing trans_e1_outcore', *time0)

    # fmmm, ftrans, fdrv for level 1
    fmmm = libmcscf.AO2MOmmm_ket_nr_s2
    ftrans = libmcscf.AO2MOtranse1_nr_s4
    fdrv = libmcscf.AO2MOnr_e2_drv
    for istep,sh_range in enumerate(shranges):
        log.debug('[%d/%d], AO [%d:%d], len(buf) = %d',
                  istep+1, nstep, *sh_range)
        buf = bufs1[:sh_range[2]]
        _ao2mo.nr_e1fill(intor, sh_range,
                         mol._atm, mol._bas, mol._env, 's4', 1, ao2mopt, buf)
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('AO integrals buffer', *ti0)
        bufpa = bufs2[:sh_range[2]]
        _ao2mo.nr_e1(buf, mo, pashape, 's4', 's1', out=bufpa)
# jc_pp, kc_pp
        if level == 1: # ppaa, papa and vhf, jcp, kcp
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('buffer-pa', *ti1)
            buf1 = bufs3[:sh_range[2]]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 buf.ctypes.data_as(ctypes.c_void_p),
                 mo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(sh_range[2]), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nao, 0, ncore),
                 ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_int(0))
            p0 = 0
            for ij in range(sh_range[0], sh_range[1]):
                i,j = lib.index_tril_to_pair(ij)
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
                    mo1 = mo_c[i0:i1]
                    tmp = numpy.einsum('uvpc,pc->uvc', buf, mo[:,:ncore])
                    tmp = lib.dot(mo1.T, tmp.reshape(di,-1))
                    j_pc += numpy.einsum('vp,pvc->pc', mo1, tmp.reshape(nmo,di,ncore))
                    tmp = numpy.einsum('uvpc,uc->vcp', buf, mo1[:,:ncore])
                    tmp = lib.dot(tmp.reshape(-1,nmo), mo).reshape(di,ncore,nmo)
                    k_pc += numpy.einsum('vp,vcp->pc', mo1, tmp)
                else:
                    dij = di * dj
                    buf = buf1[p0:p0+dij].reshape(di,dj,nao,ncore)
                    mo1 = mo_c[i0:i1]
                    mo2 = mo_c[j0:j1]
                    tmp = numpy.einsum('uvpc,pc->uvc', buf, mo[:,:ncore])
                    tmp = lib.dot(mo1.T, tmp.reshape(di,-1))
                    j_pc += numpy.einsum('vp,pvc->pc',
                                         mo2, tmp.reshape(nmo,dj,ncore)) * 2
                    tmp = numpy.einsum('uvpc,uc->vcp', buf, mo1[:,:ncore])
                    tmp = lib.dot(tmp.reshape(-1,nmo), mo).reshape(dj,ncore,nmo)
                    k_pc += numpy.einsum('vp,vcp->pc', mo2, tmp)
                    tmp = numpy.einsum('uvpc,vc->ucp', buf, mo2[:,:ncore])
                    tmp = lib.dot(tmp.reshape(-1,nmo), mo).reshape(di,ncore,nmo)
                    k_pc += numpy.einsum('up,ucp->pc', mo1, tmp)
                p0 += dij
            if log.verbose >= logger.DEBUG1:
                ti1 = log.timer('j_cp and k_cp', *ti1)

        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('half transformation of the buffer', *ti1)

# ppaa, papa
        faapp_buf[str(istep)] = \
                bufpa.reshape(sh_range[2],nmo,ncas)[:,ncore:nocc].reshape(-1,ncas**2).T
        p0 = 0
        for ij in range(sh_range[0], sh_range[1]):
            i,j = lib.index_tril_to_pair(ij)
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
            else:
                dij = di * dj
                buf1 = bufpa[p0:p0+dij].reshape(di,dj,-1)
                mo1 = mo[j0:j1,ncore:nocc].copy()
                for i in range(di):
                     lib.dot(mo1.T, buf1[i], 1, papa_buf[i0+i], 1)
            mo1 = mo[i0:i1,ncore:nocc].copy()
            buf1 = lib.dot(mo1.T, buf1.reshape(di,-1))
            papa_buf[j0:j1] += buf1.reshape(ncas,dj,-1).transpose(1,0,2)
            p0 += dij
        if log.verbose >= logger.DEBUG1:
            ti1 = log.timer('ppaa and papa buffer', *ti1)

        ti0 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)
    buf = buf1 = bufpa = None
    bufs1 = bufs2 = bufs3 = None
    time1 = log.timer('mc_ao2mo pass 1', *time0)

    log.debug1('Half transformation done. Current memory %d',
               lib.current_memory()[0])

    nblk = int(max(8, min(nmo, (max_memory*1e6/8-papa_buf.size)/(ncas**2*nmo))))
    log.debug1('nblk for papa = %d', nblk)
    dset = feri.create_dataset('papa', (nmo,ncas,nmo,ncas), 'f8')
    for i0, i1 in prange(0, nmo, nblk):
        tmp = lib.dot(mo[:,i0:i1].T, papa_buf.reshape(nao,-1))
        dset[i0:i1] = tmp.reshape(i1-i0,ncas,nmo,ncas)
    papa_buf = tmp = None
    time1 = log.timer('papa pass 2', *time1)

    tmp = numpy.empty((ncas**2,nao_pair))
    p0 = 0
    for istep, sh_range in enumerate(shranges):
        tmp[:,p0:p0+sh_range[2]] = faapp_buf[str(istep)]
        p0 += sh_range[2]
    nblk = int(max(8, min(nmo, (max_memory*1e6/8-tmp.size)/(ncas**2*nmo)-1)))
    log.debug1('nblk for ppaa = %d', nblk)
    dset = feri.create_dataset('ppaa', (nmo,nmo,ncas,ncas), 'f8')
    for i0, i1 in prange(0, nmo, nblk):
        tmp1 = _ao2mo.nr_e2(tmp, mo, (i0,i1,0,nmo), 's4', 's1', ao_loc=ao_loc)
        tmp1 = tmp1.reshape(ncas,ncas,i1-i0,nmo)
        for j in range(i1-i0):
            dset[i0+j] = tmp1[:,:,j].transpose(2,0,1)
    tmp = tmp1 = None
    time1 = log.timer('ppaa pass 2', *time1)

    time0 = log.timer('mc_ao2mo', *time0)
    return j_pc, k_pc


# level = 1: ppaa, papa and vhf, jpc, kpc
# level = 2: ppaa, papa, vhf,  jpc=0, kpc=0
class _ERIS(object):
    def __init__(self, casscf, mo, method='incore', level=1):
        mol = casscf.mol
        nao, nmo = mo.shape
        ncore = casscf.ncore
        ncas = casscf.ncas

        dm_core = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)
        vj, vk = casscf._scf.get_jk(mol, dm_core)
        self.vhf_c = reduce(numpy.dot, (mo.T, vj*2-vk, mo))

        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        eri = casscf._scf._eri
        if (method == 'incore' and eri is not None and
            (mem_incore+mem_now < casscf.max_memory*.9) or
            mol.incore_anyway):
            if eri is None:
                eri = mol.intor('int2e', aosym='s8')
            self.j_pc, self.k_pc, self.ppaa, self.papa = \
                    trans_e1_incore(eri, mo, ncore, ncas)
        else:
            import gc
            gc.collect()
            log = logger.Logger(casscf.stdout, casscf.verbose)
            self.feri = lib.H5TmpFile()
            max_memory = max(3000, casscf.max_memory*.9-mem_now)
            if max_memory < mem_basic:
                log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                         (mem_basic+mem_now)/.9, casscf.max_memory)
            self.j_pc, self.k_pc = \
                    trans_e1_outcore(mol, mo, ncore, ncas, self.feri,
                                     max_memory=max_memory,
                                     level=level, verbose=log)
            self.ppaa = self.feri['ppaa']
            self.papa = self.feri['papa']

def _mem_usage(ncore, ncas, nmo):
    nvir = nmo - ncore
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    return incore, outcore, basic

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

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
    print('j_pc ', numpy.allclose(eris0.j_pc , eris1.j_pc ))
    print('k_pc ', numpy.allclose(eris0.k_pc , eris1.k_pc ))
    print('ppaa ', numpy.allclose(eris0.ppaa , eris1.ppaa ))
    print('papa ', numpy.allclose(eris0.papa , eris1.papa ))

    print('vhf_c', numpy.allclose(eris0.vhf_c, eris2.vhf_c))
    print('j_pc ', numpy.allclose(eris0.j_pc , eris2.j_pc ))
    print('k_pc ', numpy.allclose(eris0.k_pc , eris2.k_pc ))
    print('ppaa ', numpy.allclose(eris0.ppaa , eris2.ppaa ))
    print('papa ', numpy.allclose(eris0.papa , eris2.papa ))

    print('vhf_c', numpy.allclose(eris0.vhf_c, eris3.vhf_c))
    print('ppaa ', numpy.allclose(eris0.ppaa , eris3.ppaa ))
    print('papa ', numpy.allclose(eris0.papa , eris3.papa ))

    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    eri = ao2mo.incore.full(m._eri, mo, compact=False).reshape((nmo,)*4)
    aaap = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
    ppaa = numpy.array(eri[:,:,ncore:nocc,ncore:nocc])
    papa = numpy.array(eri[:,ncore:nocc,:,ncore:nocc])
    jc_pp = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
    kc_pp = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
    vhf_c = numpy.einsum('cij->ij', jc_pp)*2 - numpy.einsum('cij->ij', kc_pp)
    j_pc = numpy.einsum('ijj->ji', jc_pp)
    k_pc = numpy.einsum('ijj->ji', kc_pp)

    print('vhf_c', numpy.allclose(vhf_c, eris1.vhf_c))
    print('j_pc ', numpy.allclose(j_pc, eris1.j_pc))
    print('k_pc ', numpy.allclose(k_pc, eris1.k_pc))
    print('ppaa ', numpy.allclose(ppaa , eris0.ppaa ))
    print('papa ', numpy.allclose(papa , eris0.papa ))

