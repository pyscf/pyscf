#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Zhi-Hao Cui <zhcui0408@gmail.com>
#

"""
Unrestricted DFT+U with kpoint sampling.
Based on KUHF routine.

Refs: PRB, 1998, 57, 1505.
"""

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft.krkspu import set_U, make_minao_lo, mdot, KRKSpU
from pyscf.data.nist import HARTREE2EV

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional + (Hubbard - double counting) for KUKSpU.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts

    # J + V_xc
    vxc = kuks.get_veff(ks, cell, dm, dm_last=dm_last, vhf_last=vhf_last,
                        hermi=hermi, kpts=kpts, kpts_band=kpts_band)
    vxc = _add_Vhubbard(vxc, ks, dm, kpts)
    return vxc

def _add_Vhubbard(vxc, ks, dm, kpts):
    '''Add Hubbard U to Vxc matrix inplace.
    '''
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    nkpts = len(kpts)
    nlo = C_ao_lo.shape[-1]

    rdm1_lo  = np.zeros((2, nkpts, nlo, nlo), dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            C_inv = np.dot(C_ao_lo[s, k].conj().T, ovlp[k])
            rdm1_lo[s, k] = mdot(C_inv, dm[s][k], C_inv.conj().T)

    is_ibz = hasattr(kpts, "kpts_ibz")
    if is_ibz:
        rdm1_lo_0 = kpts.dm_at_ref_cell(rdm1_lo)

    alphas = ks.alpha
    if not hasattr(alphas, '__len__'): # not a list or tuple
        alphas = [alphas] * len(ks.U_idx)

    E_U = 0.0
    weight = getattr(kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab, alpha in zip(ks.U_idx, ks.U_val, ks.U_lab, alphas):
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
                    E_U += weight[k] * (val * 0.5) * (P_k.trace() - np.dot(P_k, P_k).trace())
                    vhub_loc = (np.eye(P_k.shape[-1]) - P_k * 2.0) * (val * 0.5)
                    if alpha is not None:
                        # The alpha perturbation is only applied to the linear term of
                        # the local density.
                        E_U += weight[k] * alpha * P_k.trace()
                        vhub_loc += np.eye(P_k.shape[-1]) * alpha
                    SC = np.dot(S_k, C_k)
                    vhub_loc = SC.dot(vhub_loc).dot(SC.conj().T)
                    if vxc.dtype == np.float64:
                        vhub_loc = vhub_loc.real
                    vxc[s,k] += vhub_loc
                    if not is_ibz:
                        P_loc += P_k
                if is_ibz:
                    P_loc = rdm1_lo_0[s][U_mesh].real
                else:
                    P_loc = P_loc.real / nkpts
                logger.info(ks, "spin %s\n%s\n%s", s, lab_string, P_loc)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%g) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U.real)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    """
    Electronic energy for KUKSpU.
    """
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = getattr(mf.kpts, "weights_ibz",
                     np.array([1.0/len(h1e_kpts),]*len(h1e_kpts)))
    e1 = (np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[0]) +
          np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[1]))
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    mf.scf_summary['E_U'] = vhf.E_U.real

    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s',
                 e1, vhf.ecoul, vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

class KUKSpU(kuks.KUKS):
    """
    UKSpU class adapted for PBCs with k-point sampling.
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 'alpha'}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = lib.invalid_method('to_hf')

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO', **kwargs):
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
                     np.array, shape ((spin,), nkpts, nao, nlo),
                     string, in 'minao'.
            minao_ref: reference for minao orbitals, default is 'MINAO'.

        Attributes:
            U_idx: same as the input.
            U_val: effectiv U-J [in AU]
            C_ao_loc: np.array
            alpha: the perturbation [in AU] used to compute U in LR-cDFT.
                Refs: Cococcioni and de Gironcoli, PRB 71, 035105 (2005)
        """
        super(self.__class__, self).__init__(cell, kpts, xc=xc, exxdiv=exxdiv, **kwargs)

        set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)
        if self.C_ao_lo.ndim == 3:
            self.C_ao_lo = np.asarray((self.C_ao_lo, self.C_ao_lo))
        elif self.C_ao_lo.ndim == 4:
            if self.C_ao_lo.shape[0] == 1:
                self.C_ao_lo = np.asarray((self.C_ao_lo[0], self.C_ao_lo[0]))
            assert self.C_ao_lo.shape[0] == 2
        else:
            raise ValueError

        # The perturbation (eV) used to compute U in LR-cDFT.
        self.alpha = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.INFO:
            from pyscf.pbc.dft.krkspu import _print_U_info
            _print_U_info(self, log)
        return self

    def nuc_grad_method(self):
        raise NotImplementedError

def linear_response_u(mf_plus_u, alphalist=(0.02, 0.05, 0.08)):
    '''
    Refs:
        [1] M. Cococcioni and S. de Gironcoli, Phys. Rev. B 71, 035105 (2005)
        [2] H. J. Kulik, M. Cococcioni, D. A. Scherlis, and N. Marzari, Phys. Rev. Lett. 97, 103001 (2006)
        [3] Heather J. Kulik, J. Chem. Phys. 142, 240901 (2015)
        [4] https://hjkgrp.mit.edu/tutorials/2011-05-31-calculating-hubbard-u/
        [5] https://hjkgrp.mit.edu/tutorials/2011-06-28-hubbard-u-multiple-sites/

    Args:
        alphalist :
            alpha parameters (in eV) are the displacements for the linear
            response calculations. For each alpha in this list, the DFT+U with
            U=u0+alpha, U=u0-alpha are evaluated. u0 is the U value from the
            reference mf_plus_u object, which will be treated as a standard DFT
            functional.
    '''
    is_ibz = hasattr(mf_plus_u.kpts, "kpts_ibz")
    if is_ibz:
        raise NotImplementedError

    assert isinstance(mf_plus_u, KUKSpU)
    assert len(mf_plus_u.U_idx) > 0
    if not mf_plus_u.converged:
        mf_plus_u.run()
    assert mf_plus_u.converged
    # The bare density matrix without adding U
    bare_dm = mf_plus_u.make_rdm1()

    mf = mf_plus_u.copy()
    log = logger.new_logger(mf)

    alphalist = np.asarray(alphalist)
    alphalist = np.append(-alphalist[::-1], alphalist)

    nkpts = len(mf.kpts)
    C_ao_lo = mf.C_ao_lo
    ovlp = mf.get_ovlp()
    C_inv = []
    for local_idx in mf.U_idx:
        C_inv.append(
            [[C_ao_lo[0,k][:,local_idx].conj().T.dot(ovlp[k]) for k in range(nkpts)],
             [C_ao_lo[1,k][:,local_idx].conj().T.dot(ovlp[k]) for k in range(nkpts)]])

    bare_occupancies = []
    final_occupancies = []
    for alpha in alphalist:
        mf.alpha = alpha / HARTREE2EV
        mf.kernel(dm0=bare_dm)
        local_occ = 0
        for c in C_inv:
            C_on_site = [[c[0][k].dot(mf.mo_coeff[0][k]) for k in range(nkpts)],
                         [c[1][k].dot(mf.mo_coeff[1][k]) for k in range(nkpts)]]
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += sum(x.trace().real for x in rdm1_lo[0])
            local_occ += sum(x.trace().real for x in rdm1_lo[1])
        local_occ /= nkpts
        final_occupancies.append(local_occ)

        # The first iteration of SCF
        fock = mf.get_fock(dm=bare_dm)
        e, mo = mf.eig(fock, ovlp)
        local_occ = 0
        for c in C_inv:
            C_on_site = [[c[0][k].dot(mo[0][k]) for k in range(nkpts)],
                         [c[1][k].dot(mo[1][k]) for k in range(nkpts)]]
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += sum(x.trace().real for x in rdm1_lo[0])
            local_occ += sum(x.trace().real for x in rdm1_lo[1])
        local_occ /= nkpts
        bare_occupancies.append(local_occ)
        log.info('alpha=%f bare_occ=%g final_occ=%g',
                 alpha, bare_occupancies[-1], final_occupancies[-1])

    chi0, occ0 = np.polyfit(alphalist, bare_occupancies, deg=1)
    chif, occf = np.polyfit(alphalist, final_occupancies, deg=1)
    log.info('Line fitting chi0 = %f x + %f', chi0, occ0)
    log.info('Line fitting chif = %f x + %f', chif, occf)
    Uresp = 1./chi0 - 1./chif
    log.note('Uresp = %f, chi0 = %f, chif = %f', Uresp, chi0, chif)
    return Uresp
