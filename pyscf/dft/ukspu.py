#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
DFT+U on molecules

See also the pbc.dft.krkspu and pbc.dft.kukspu module
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.dft import uks
from pyscf.dft.rkspu import set_U, make_minao_lo

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """
    Coulomb + XC functional + (Hubbard - double counting) for UKS+U.
    """
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    # J + V_xc
    vxc = super(ks.__class__, ks).get_veff(
        mol, dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    C_inv = np.dot(C_ao_lo.conj().T, ovlp)
    rdm1_lo = [C_inv.dot(dm[0]).dot(C_inv.conj().T),
               C_inv.dot(dm[1]).dot(C_inv.conj().T)]

    E_U = 0.0
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            for s in range(2):
                P = rdm1_lo[s][idx[:,None], idx]
                SC = np.dot(ovlp, C_ao_lo[:, idx])
                loc_sites = P.shape[-1]
                vxc[s] += SC.dot((np.eye(loc_sites) - P * 2.0) * (val * 0.5)).dot(SC.conj().T)
                E_U += (val * 0.5) * (P.trace() - np.dot(P, P).trace())
                logger.info(ks, "spin %s\n%s\n%s", s, lab_string, P)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U.real)
    return vxc

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    """
    Electronic energy for UKSpU.
    """
    if h1e is None: h1e = mf.get_hcore()
    if dm is None: dm = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.mol, dm)

    e1 = (np.einsum('ij,ji->', h1e, dm[0]) +
          np.einsum('ij,ji->', h1e, dm[1]))
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    mf.scf_summary['E_U'] = vhf.E_U.real

    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s',
                 e1, vhf.ecoul, vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

class UKSpU(uks.UKS):
    """
    UKSpU class adapted for PBCs with k-point sampling.
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab"}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = lib.invalid_method('to_hf')

    def __init__(self, mol, xc='LDA,VWN',
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO'):
        """
        DFT+U args:
            U_idx: can be
                   list of list: each sublist is a set of LO indices to add U.
                   list of string: each string is one kind of LO orbitals,
                                   e.g. ['Ni 3d', '1 O 2pz'], in this case,
                                   LO should be aranged as ao_labels order.
                   or a combination of these two.
            U_val: a list of effective U [in eV], i.e. U-J in Dudarev's DFT+U.
                   each U corresponds to one kind of LO orbitals, should have
                   the same length as U_idx.
            C_ao_lo: LO coefficients, can be
                     np.array, shape ((spin,), nao, nlo),
                     string, in 'minao'.
            minao_ref: reference for minao orbitals, default is 'MINAO'.
        """
        super(self.__class__, self).__init__(mol, xc=xc)

        set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
        assert self.C_ao_lo.ndim == 2

    def nuc_grad_method(self):
        raise NotImplementedError
