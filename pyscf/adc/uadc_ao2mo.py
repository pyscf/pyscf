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
import h5py
import tempfile

### Integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]
    ind_vv_g = np.tril_indices(nvir_a, k=-1)
    ind_VV_g = np.tril_indices(nvir_b, k=-1)
   
    eris = lambda:None

    # TODO: check if myadc._scf._eri is not None
    eris.oooo = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_a, occ_a), compact=False).reshape(nocc_a, nocc_a, nocc_a, nocc_a).copy()
    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nocc_a, nocc_a).copy()
    eris.ovov = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_a, vir_a), compact=False).reshape(nocc_a, nvir_a, nocc_a, nvir_a).copy()
    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nvir_a, nocc_a).copy()
    eris.oovv = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_a, vir_a), compact=False).reshape(nocc_a, nocc_a, nvir_a, nvir_a).copy()
    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, vir_a), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OOOO = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, occ_b, occ_b), compact=False).reshape(nocc_b, nocc_b, nocc_b, nocc_b).copy()
    eris.OVOO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nocc_b, nocc_b).copy()
    eris.OVOV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_b, vir_b), compact=False).reshape(nocc_b, nvir_b, nocc_b, nvir_b).copy()
    eris.OOVV = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_b, vir_b), compact=False).reshape(nocc_b, nocc_b, nvir_b, nvir_b).copy()
    eris.OVVO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nvir_b, nocc_b).copy()
    eris.OVVV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, vir_b), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    eris.ooOO = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_b, occ_b), compact=False).reshape(nocc_a, nocc_a, nocc_b, nocc_b).copy()
    eris.ovOO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nocc_b, nocc_b).copy()
    eris.ovOV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_b, vir_b), compact=False).reshape(nocc_a, nvir_a, nocc_b, nvir_b).copy()
    eris.ooVV = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_b, vir_b), compact=False).reshape(nocc_a, nocc_a, nvir_b, nvir_b).copy()
    eris.ovVO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nvir_b, nocc_b).copy()
    eris.ovVV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, vir_b), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OVoo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nocc_a, nocc_a).copy()
    eris.OOvv = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_a, vir_a), compact=False).reshape(nocc_b, nocc_b, nvir_a, nvir_a).copy()
    eris.OVov = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_a, vir_a), compact=False).reshape(nocc_b, nvir_b, nocc_a, nvir_a).copy()
    eris.OVvo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nvir_a, nocc_a).copy()
    eris.OVvv = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, vir_a), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

        eris.vvvv_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_a, vir_a), compact=False).reshape(nvir_a, nvir_a, nvir_a, nvir_a)
        eris.vvvv_p = eris.vvvv_p.transpose(0,2,1,3)
        eris.vvvv_p -= eris.vvvv_p.transpose(0,1,3,2)
        eris.vvvv_p = eris.vvvv_p[:, :, ind_vv_g[0], ind_vv_g[1]]
        eris.vvvv_p = eris.vvvv_p[ind_vv_g[0], ind_vv_g[1]].copy()

        eris.VVVV_p = ao2mo.general(myadc._scf._eri, (vir_b, vir_b, vir_b, vir_b), compact=False).reshape(nvir_b, nvir_b, nvir_b, nvir_b)
        eris.VVVV_p = eris.VVVV_p.transpose(0,2,1,3)
        eris.VVVV_p -= eris.VVVV_p.transpose(0,1,3,2)
        eris.VVVV_p = eris.VVVV_p[:, :, ind_VV_g[0], ind_VV_g[1]]
        eris.VVVV_p = eris.VVVV_p[ind_VV_g[0], ind_VV_g[1]].copy()

        eris.vVvV_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_b, vir_b), compact=False).reshape(nvir_a, nvir_a, nvir_b, nvir_b)
        eris.vVvV_p = np.ascontiguousarray(eris.vVvV_p.transpose(0,2,1,3)) 
        eris.vVvV_p = eris.vVvV_p.reshape(nvir_a*nvir_b, nvir_a*nvir_b) 

    return eris


def transform_integrals_outcore(myadc):

    #cput0 = (time.clock(), time.time())
    #log = logger.Logger(myadc.stdout, myadc.verbose)
    #mol = myadc.mol
    #mo_coeff = myadc.mo_coeff
    #nao = mo_coeff.shape[0]
    #nmo = myadc._nmo

    mo_a = myadc.mo_coeff[0]
    mo_b = myadc.mo_coeff[1]
    nmo_a = mo_a.shape[1]
    nmo_b = mo_b.shape[1]

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]

    nvpair_a = nvir_a * (nvir_a+1) // 2
    nvpair_b = nvir_b * (nvir_b+1) // 2

    ind_vv_g = np.tril_indices(nvir_a, k=-1)
    ind_VV_g = np.tril_indices(nvir_b, k=-1)

    eris = lambda:None

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc_a,nocc_a,nocc_a,nocc_a), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc_a,nocc_a,nvir_a,nvir_a), 'f8', chunks=(nocc_a,nocc_a,1,nvir_a))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc_a,nvir_a,nocc_a,nocc_a), 'f8', chunks=(nocc_a,1,nocc_a,nocc_a))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc_a,nvir_a,nvir_a,nocc_a), 'f8', chunks=(nocc_a,1,nvir_a,nocc_a))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc_a,nvir_a,nocc_a,nvir_a), 'f8', chunks=(nocc_a,1,nocc_a,nvir_a))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc_a,nvir_a,nvpair_a), 'f8')


    eris.OOOO = eris.feri1.create_dataset('OOOO', (nocc_b,nocc_b,nocc_b,nocc_b), 'f8')
    eris.OOVV = eris.feri1.create_dataset('OOVV', (nocc_b,nocc_b,nvir_b,nvir_b), 'f8', chunks=(nocc_b,nocc_b,1,nvir_b))
    eris.OVOO = eris.feri1.create_dataset('OVOO', (nocc_b,nvir_b,nocc_b,nocc_b), 'f8', chunks=(nocc_b,1,nocc_b,nocc_b))
    eris.OVVO = eris.feri1.create_dataset('OVVO', (nocc_b,nvir_b,nvir_b,nocc_b), 'f8', chunks=(nocc_b,1,nvir_b,nocc_b))
    eris.OVOV = eris.feri1.create_dataset('OVOV', (nocc_b,nvir_b,nocc_b,nvir_b), 'f8', chunks=(nocc_b,1,nocc_b,nvir_b))
    eris.OVVV = eris.feri1.create_dataset('OVVV', (nocc_b,nvir_b,nvpair_b), 'f8')


    eris.ooOO = eris.feri1.create_dataset('ooOO', (nocc_a,nocc_a,nocc_b,nocc_b), 'f8')
    eris.ooVV = eris.feri1.create_dataset('ooVV', (nocc_a,nocc_a,nvir_b,nvir_b), 'f8', chunks=(nocc_a,nocc_a,1,nvir_b))
    eris.ovOO = eris.feri1.create_dataset('ovOO', (nocc_a,nvir_a,nocc_b,nocc_b), 'f8', chunks=(nocc_a,1,nocc_b,nocc_b))
    eris.ovVO = eris.feri1.create_dataset('ovVO', (nocc_a,nvir_a,nvir_b,nocc_b), 'f8', chunks=(nocc_a,1,nvir_b,nocc_b))
    eris.ovOV = eris.feri1.create_dataset('ovOV', (nocc_a,nvir_a,nocc_b,nvir_b), 'f8', chunks=(nocc_a,1,nocc_b,nvir_b))
    eris.ovVV = eris.feri1.create_dataset('ovVV', (nocc_a,nvir_a,nvpair_b), 'f8')


    eris.OOvv = eris.feri1.create_dataset('OOvv', (nocc_b,nocc_b,nvir_a,nvir_a), 'f8', chunks=(nocc_b,nocc_b,1,nvir_a))
    eris.OVoo = eris.feri1.create_dataset('OVoo', (nocc_b,nvir_b,nocc_a,nocc_a), 'f8', chunks=(nocc_b,1,nocc_a,nocc_a))
    eris.OVvo = eris.feri1.create_dataset('OVvo', (nocc_b,nvir_b,nvir_a,nocc_a), 'f8', chunks=(nocc_b,1,nvir_a,nocc_a))
    eris.OVov = eris.feri1.create_dataset('OVov', (nocc_b,nvir_b,nocc_a,nvir_a), 'f8', chunks=(nocc_b,1,nocc_a,nvir_a))
    eris.OVvv = eris.feri1.create_dataset('OVvv', (nocc_b,nvir_b,nvpair_a), 'f8')


    cput1 = time.clock(), time.time()
    mol = myadc.mol
    # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
    tmpf = lib.H5TmpFile()
    if nocc_a > 0:
        ao2mo.general(mol, (occ_a,mo_a,mo_a,mo_a), tmpf, 'aa')
        buf = np.empty((nmo_a,nmo_a,nmo_a))
        for i in range(nocc_a):
            lib.unpack_tril(tmpf['aa'][i*nmo_a:(i+1)*nmo_a], out=buf)
            eris.oooo[i] = buf[:nocc_a,:nocc_a,:nocc_a]
            eris.ovoo[i] = buf[nocc_a:,:nocc_a,:nocc_a]
            eris.ovov[i] = buf[nocc_a:,:nocc_a,nocc_a:]
            eris.oovv[i] = buf[:nocc_a,nocc_a:,nocc_a:]
            eris.ovvo[i] = buf[nocc_a:,nocc_a:,:nocc_a]
            eris.ovvv[i] = lib.pack_tril(buf[nocc_a:,nocc_a:,nocc_a:])
        del(tmpf['aa'])

    if nocc_b > 0:
        buf = np.empty((nmo_b,nmo_b,nmo_b))
        ao2mo.general(mol, (occ_b,mo_b,mo_b,mo_b), tmpf, 'bb')
        for i in range(nocc_b):
            lib.unpack_tril(tmpf['bb'][i*nmo_b:(i+1)*nmo_b], out=buf)
            eris.OOOO[i] = buf[:nocc_b,:nocc_b,:nocc_b]
            eris.OVOO[i] = buf[nocc_b:,:nocc_b,:nocc_b]
            eris.OVOV[i] = buf[nocc_b:,:nocc_b,nocc_b:]
            eris.OOVV[i] = buf[:nocc_b,nocc_b:,nocc_b:]
            eris.OVVO[i] = buf[nocc_b:,nocc_b:,:nocc_b]
            eris.OVVV[i] = lib.pack_tril(buf[nocc_b:,nocc_b:,nocc_b:])
        del(tmpf['bb'])

    if nocc_a > 0:
        buf = np.empty((nmo_a,nmo_b,nmo_b))
        ao2mo.general(mol, (occ_a,mo_a,mo_b,mo_b), tmpf, 'ab')
        for i in range(nocc_a):
            lib.unpack_tril(tmpf['ab'][i*nmo_a:(i+1)*nmo_a], out=buf)
            eris.ooOO[i] = buf[:nocc_a,:nocc_b,:nocc_b]
            eris.ovOO[i] = buf[nocc_a:,:nocc_b,:nocc_b]
            eris.ovOV[i] = buf[nocc_a:,:nocc_b,nocc_b:]
            eris.ooVV[i] = buf[:nocc_a,nocc_b:,nocc_b:]
            eris.ovVO[i] = buf[nocc_a:,nocc_b:,:nocc_b]
            eris.ovVV[i] = lib.pack_tril(buf[nocc_a:,nocc_b:,nocc_b:])
        del(tmpf['ab'])

    if nocc_b > 0:
        buf = np.empty((nmo_b,nmo_a,nmo_a))
        ao2mo.general(mol, (occ_b,mo_b,mo_a,mo_a), tmpf, 'ba')
        for i in range(nocc_b):
            lib.unpack_tril(tmpf['ba'][i*nmo_b:(i+1)*nmo_b], out=buf)
            eris.OVoo[i] = buf[nocc_b:,:nocc_a,:nocc_a]
            eris.OVov[i] = buf[nocc_b:,:nocc_a,nocc_a:]
            eris.OOvv[i] = buf[:nocc_b,nocc_a:,nocc_a:]
            eris.OVvo[i] = buf[nocc_b:,nocc_a:,:nocc_a]
            eris.OVvv[i] = lib.pack_tril(buf[nocc_b:,nocc_a:,nocc_a:])
        del(tmpf['ba'])

    buf = None
    cput1 = logger.timer_debug1(myadc, 'transforming oopq, ovpq', *cput1)

################## forming eris_vvvv ########################################

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

        eris.vvvv_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_a, vir_a), compact=False).reshape(nvir_a, nvir_a, nvir_a, nvir_a)
        eris.vvvv_p = eris.vvvv_p.transpose(0,2,1,3)
        eris.vvvv_p -= eris.vvvv_p.transpose(0,1,3,2)
        eris.vvvv_p = eris.vvvv_p[:, :, ind_vv_g[0], ind_vv_g[1]]
        eris.vvvv_p = eris.vvvv_p[ind_vv_g[0], ind_vv_g[1]].copy()

        eris.VVVV_p = ao2mo.general(myadc._scf._eri, (vir_b, vir_b, vir_b, vir_b), compact=False).reshape(nvir_b, nvir_b, nvir_b, nvir_b)
        eris.VVVV_p = eris.VVVV_p.transpose(0,2,1,3)
        eris.VVVV_p -= eris.VVVV_p.transpose(0,1,3,2)
        eris.VVVV_p = eris.VVVV_p[:, :, ind_VV_g[0], ind_VV_g[1]]
        eris.VVVV_p = eris.VVVV_p[ind_VV_g[0], ind_VV_g[1]].copy()

        eris.vVvV_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_b, vir_b), compact=False).reshape(nvir_a, nvir_a, nvir_b, nvir_b)
        eris.vVvV_p = np.ascontiguousarray(eris.vVvV_p.transpose(0,2,1,3)) 
        eris.vVvV_p = eris.vVvV_p.reshape(nvir_a*nvir_b, nvir_a*nvir_b) 

    return eris


def dataset(data):
    _, fname = tempfile.mkstemp()
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data)
