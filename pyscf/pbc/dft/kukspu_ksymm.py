#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.dft import kukspu, kuks_ksymm
from pyscf.pbc.lib import kpts as libkpts

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + (Hubbard - double counting) for KUKSpU.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    if isinstance(kpts, np.ndarray):
        return kukspu.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)

    # J + V_xc
    vxc = kuks_ksymm.get_veff(ks, cell, dm, dm_last=dm_last, vhf_last=vhf_last,
                              hermi=hermi, kpts=kpts, kpts_band=kpts_band)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nkpts = len(kpts)
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((2, nkpts, nlo, nlo), dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            C_inv = np.dot(C_ao_lo[s, k].conj().T, ovlp[k])
            rdm1_lo[s, k] = C_inv.dot(dm[s][k]).dot(C_inv.conj().T)
    rdm1_lo_0 = kpts.dm_at_ref_cell(rdm1_lo)

    E_U = 0.0
    weight = kpts.weights_ibz
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            U_mesh = np.ix_(idx, idx)
            for s in range(2):
                P_loc = 0.0
                for k in range(nkpts):
                    S_k = ovlp[k]
                    C_k = C_ao_lo[s, k][:, idx]
                    P_k = rdm1_lo[s, k][U_mesh]
                    SC = np.dot(S_k, C_k)
                    vhub_loc = (np.eye(P_k.shape[-1]) - P_k * 2.0) * (val * 0.5)
                    vxc[s][k] += SC.dot(vhub_loc).dot(SC.conj().T).astype(vxc[s][k].dtype,copy=False)
                    E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.dot(P_k, P_k).trace())
                P_loc = rdm1_lo_0[s][U_mesh].real
                logger.info(ks, "spin %s\n%s\n%s", s, lab_string, P_loc)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U.real)
    return vxc

@lib.with_doc(kukspu.KUKSpU.__doc__)
class KsymAdaptedKUKSpU(kuks_ksymm.KUKS):

    get_veff = get_veff
    energy_elec = kukspu.energy_elec
    to_hf = lib.invalid_method('to_hf')

    @lib.with_doc(kukspu.KUKSpU.__init__.__doc__)
    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO', **kwargs):
        kukspu.KUKSpU.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv,
                               U_idx=U_idx, U_val=U_val, C_ao_lo=C_ao_lo,
                               minao_ref=minao_ref, **kwargs)

KUKSpU = KsymAdaptedKUKSpU
