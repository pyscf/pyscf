# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
import time

### Integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

    occ = myadc.mo_coeff[:,:myadc._nocc]
    vir = myadc.mo_coeff[:,myadc._nocc:]

    nocc = occ.shape[1]
    nvir = vir.shape[1]

    eris = lambda:None

    # TODO: check if myadc._scf._eri is not None

    eris.oooo = ao2mo.general(myadc._scf._eri, (occ, occ, occ, occ), compact=False).reshape(nocc, nocc, nocc, nocc).copy()
    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ, vir, occ, occ), compact=False).reshape(nocc, nvir, nocc, nocc).copy()
    eris.ovov = ao2mo.general(myadc._scf._eri, (occ, vir, occ, vir), compact=False).reshape(nocc, nvir, nocc, nvir).copy()
    eris.oovv = ao2mo.general(myadc._scf._eri, (occ, occ, vir, vir), compact=False).reshape(nocc, nocc, nvir, nvir).copy()
    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ, vir, vir, occ), compact=False).reshape(nocc, nvir, nvir, nocc).copy()
    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ, vir, vir, vir), compact=True).reshape(nocc, nvir, -1).copy()

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
        eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3)) 
        eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)

    return eris

#@profile    
def transform_integrals_outcore(myadc):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    mol = myadc.mol
    mo_coeff = myadc.mo_coeff
    nao = mo_coeff.shape[0]
    nmo = myadc._nmo

    occ = myadc.mo_coeff[:,:myadc._nocc]
    vir = myadc.mo_coeff[:,myadc._nocc:]

    nocc = occ.shape[1]
    nvir = vir.shape[1]
    nvpair = nvir * (nvir+1) // 2

    eris = lambda:None

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')

    #eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'f8', chunks=(nvir,1,nvir,nvir))

    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.oovv[p0:p1] = eri[:,:,nocc:,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.ovoo[:,p0:p1] = eri[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = eri[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = eri[:,:,:nocc,nocc:].transpose(1,0,2,3)
        vvv = lib.pack_tril(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.ovvv[:,p0:p1] = vvv.reshape(p1-p0,nocc,nvpair).transpose(1,0,2)

################## forming eris_vvvv ########################################
    cput1 = time.clock(), time.time()
    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
        max_memory = max(myadc.memorymin, myadc.max_memory-lib.current_memory()[0])
        eris.feri2 = lib.H5TmpFile()
        ao2mo.full(mol, vir, eris.feri2, max_memory=max_memory, verbose=log, compact=True)
        eris.vvvv = eris.feri2['eri_mo']
        cput1 = log.timer_debug1('transforming vvvv', *cput1)

        tmp = np.einsum('abcd',get_vvvv(eris))
        print (tmp.shape)

        exit()
        nvira, nvirb = eris.vvvv.shape[-2:]
        max_memory = myadc.max_memory - lib.current_memory()[0]
        #unit = nvirb**2*nvira*2 + nocc2*nvirb + 1
        unit = nvirb**2*nvira*2 + nvira*nvirb + 1
        blksize = min(nvira, max(myadc.blkmin, int(max_memory*1e6/8/unit)))
    
        for p0,p1 in lib.prange(0, nvira, blksize):
            #eris.vvvv[p0:p1] = eris.vvvv[p0:p1].transpose(0,2,1,3)
            print (eris.vvvv[p0:p1].shape)
            exit()
        #    #time0 = log.timer_debug1('vvvv [%d:%d]' % (p0,p1), *time0)
        #eris.vvvv.reshape(nvir*nvir, nvir*nvir) 
        exit()
##############################################################################

    fswap = lib.H5TmpFile()
    max_memory = max(myadc.memorymin, myadc.max_memory-lib.current_memory()[0])
    int2e = mol._add_suffix('int2e')
    ao2mo.outcore.half_e1(mol, (mo_coeff,occ), fswap, int2e,
                          's4', 1, max_memory, verbose=log)

    ao_loc = mol.ao_loc_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
    blksize = min(nmo, max(myadc.blkmin, blksize))
    log.debug1('blksize %d', blksize)
    cput2 = cput1

    fload = ao2mo.outcore._load_from_h5g
    buf = np.empty((blksize*nocc,nao_pair))
    buf_prefetch = np.empty_like(buf)
    def load(buf_prefetch, p0, rowmax):
        if p0 < rowmax:
            p1 = min(rowmax, p0+blksize)
            fload(fswap['0'], p0*nocc, p1*nocc, buf_prefetch)

    outbuf = np.empty((blksize*nocc,nmo**2))
    with lib.call_in_background(load, sync=not myadc.async_io) as prefetch:
        prefetch(buf_prefetch, 0, nocc)
        for p0, p1 in lib.prange(0, nocc, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, p1, nocc)

            nrow = (p1 - p0) * nocc
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_occ_frac(p0, p1, dat)
        cput2 = log.timer_debug1('transforming oopp', *cput2) 

        prefetch(buf_prefetch, nocc, nmo)
        for p0, p1 in lib.prange(0, nvir, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, nocc+p0, nmo)

            nrow = (p1 - p0) * nocc 
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_vir_frac(p0, p1, dat)

            cput2 = log.timer_debug1('transforming ovpp [%d:%d]'%(p0,p1), *cput2)

    cput1 = log.timer_debug1('transforming oppp', *cput1)
    log.timer('ADC integral transformation', *cput0)

    #eris.ovvv = ao2mo.general(myadc._scf._eri, (occ, vir, vir, vir), compact=True).reshape(nocc, nvir, -1).copy()

#    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
#        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
#        eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3)) 
#        eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)

    return eris


def get_vvvv(eris):
      
    nvir = int(np.sqrt(eris.vvvv.shape[0]*2))
    return ao2mo.restore(1, np.asarray(eris.vvvv), nvir)




#def get_vvvv_p(myadc, mol, vvvv, out=None, verbose=None):
#    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
#    where vvvv can be real or complex and no permutation symmetry is available in vvvv.
#
#    Args:
#        vvvv : None or integral object
#            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
#    '''
#    assert(vvvv is not None)
#
#    nvira, nvirb = vvvv.shape[-2:]
#    #x2 = t2.reshape(-1,nvira,nvirb)
#    #nocc2 = x2.shape[0]
#    #dtype = numpy.result_type(t2, vvvv)
#    #Ht2 = numpy.ndarray(x2.shape, dtype=dtype, buffer=out)
#
#    max_memory = mycc.max_memory - lib.current_memory()[0]
#    unit = nvirb**2*nvira*2 + nocc2*nvirb + 1
#    blksize = min(nvira, max(BLKMIN, int(max_memory*1e6/8/unit)))
#
#    for p0,p1 in lib.prange(0, nvira, blksize):
#        vvvv_p[:,p0:p1] = lib.einsum('acbd->abcd', vvvv[p0:p1])
#        #time0 = log.timer_debug1('vvvv [%d:%d]' % (p0,p1), *time0)
#    return vvvv_p.reshape(nvir*nvir, nvir*nvir) 


def unpack_eri_1(eri, norb):

    n_oo = norb * (norb + 1) // 2
    ind_oo = np.tril_indices(norb)

    eri_ = None

    if len(eri.shape) == 3:
        if (eri.shape[0] == n_oo):
            eri_ = np.zeros((norb, norb, eri.shape[1], eri.shape[2]))
            eri_[ind_oo[0], ind_oo[1]] = eri
            eri_[ind_oo[1], ind_oo[0]] = eri

        elif (eri.shape[2] == n_oo):
            eri_ = np.zeros((eri.shape[0], eri.shape[1], norb, norb))
            eri_[:, :, ind_oo[0], ind_oo[1]] = eri
            eri_[:, :, ind_oo[1], ind_oo[0]] = eri
        else:
            raise TypeError("ERI dimensions don't match")

    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_


def unpack_eri_2(eri, norb):

    n_oo = norb * (norb - 1) // 2
    ind_oo = np.tril_indices(norb,k=-1)

    eri_ = None

    if len(eri.shape) == 2:
        if (eri.shape[0] != n_oo or eri.shape[1] != n_oo):
            raise TypeError("ERI dimensions don't match")

        temp = np.zeros((n_oo, norb, norb))
        temp[:, ind_oo[0], ind_oo[1]] = eri
        temp[:, ind_oo[1], ind_oo[0]] = -eri
        eri_ = np.zeros((norb, norb, norb, norb))
        eri_[ind_oo[0], ind_oo[1]] = temp
        eri_[ind_oo[1], ind_oo[0]] = -temp
    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_
