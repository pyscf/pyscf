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
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
import time
import h5py
import tempfile

### Incore integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

     cput0 = (time.clock(), time.time())
     log = logger.Logger(myadc.stdout, myadc.verbose)
     cell = myadc.cell
     kpts = myadc.kpts
     nkpts = myadc.nkpts
     nocc = myadc.nocc
     nmo = myadc.nmo
     nvir = nmo - nocc

     dtype = myadc.mo_coeff[0].dtype

     mo_coeff = myadc.mo_coeff = padded_mo_coeff(myadc, myadc.mo_coeff)

     # Get location of padded elements in occupied and virtual space.
     nocc_per_kpt = get_nocc(myadc, per_kpoint=True)
     nonzero_padding = padding_k_idx(myadc, kind="joint")

     fao2mo = myadc._scf.with_df.ao2mo

     kconserv = myadc.khelper.kconserv
     khelper = myadc.khelper
     orbo = np.asarray(mo_coeff[:,:,:nocc], order='C')
     orbv = np.asarray(mo_coeff[:,:,nocc:], order='C')

     fao2mo = myadc._scf.with_df.ao2mo
     eris = lambda:None

     log.info('using incore ERI storage')
     eris.oooo = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
     eris.ooov = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir), dtype=dtype)
     eris.oovv = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
     eris.ovov = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
     eris.voov = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nocc,nvir), dtype=dtype)
     eris.vovv = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir), dtype=dtype)
     #self.vvvv = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), dtype=dtype)
     #self.vvvv = cc._scf.with_df.ao2mo_7d(orbv, factor=1./nkpts).transpose(0,2,1,3,5,4,6)

     for (ikp,ikq,ikr) in khelper.symm_map.keys():
          iks = kconserv[ikp,ikq,ikr]
          eri_kpt = fao2mo((mo_coeff[ikp],mo_coeff[ikq],mo_coeff[ikr],mo_coeff[iks]),
                           (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
          if dtype == np.float: eri_kpt = eri_kpt.real
          eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
          for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
              eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
              eris.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
              eris.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
              eris.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
              eris.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
              eris.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
              eris.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
              #self.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

     return eris



#    cput0 = (time.clock(), time.time())
#    log = logger.Logger(myadc.stdout, myadc.verbose)
#
#    occ = myadc.mo_coeff[:,:myadc._nocc]
#    vir = myadc.mo_coeff[:,myadc._nocc:]
#
#    nocc = occ.shape[1]
#    nvir = vir.shape[1]
#
#    eris = lambda:None
#
#    # TODO: check if myadc._scf._eri is not None
#
#    eris.oooo = ao2mo.general(myadc._scf._eri, (occ, occ, occ, occ), compact=False).reshape(nocc, nocc, nocc, nocc).copy()
#    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ, vir, occ, occ), compact=False).reshape(nocc, nvir, nocc, nocc).copy()
#    eris.oovv = ao2mo.general(myadc._scf._eri, (occ, occ, vir, vir), compact=False).reshape(nocc, nocc, nvir, nvir).copy()
#    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ, vir, vir, occ), compact=False).reshape(nocc, nvir, nvir, nocc).copy()
#    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ, vir, vir, vir), compact=True).reshape(nocc, nvir, -1).copy()
#
#    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
#        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
#        eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3)) 
#        eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)
#
#    log.timer('ADC integral transformation', *cput0)
#    return eris
