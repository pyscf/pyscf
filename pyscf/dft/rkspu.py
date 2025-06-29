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
DFT+U for molecules

See also the pbc.dft.krkspu and pbc.dft.kukspu module

Refs:
    Heather J. Kulik, J. Chem. Phys. 142, 240901 (2015)
'''

import numpy as np
import scipy.linalg as la
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import rks
from pyscf.data.nist import HARTREE2EV
from pyscf import lo
from pyscf.lo import iao

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """
    Coulomb + XC functional + Hubbard U terms for RKS+U.

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : ``(nao, nao)`` or ``(*, nao, nao)`` ndarray
        Veff = J + Vxc + V_U.
    """
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()

    # J + V_xc
    vxc = rks.get_veff(ks, mol, dm, dm_last=dm_last, vhf_last=vhf_last,
                       hermi=hermi)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    C_inv = np.dot(C_ao_lo.conj().T, ovlp)
    rdm1_lo = C_inv.dot(dm).dot(C_inv.conj().T)

    alphas = ks.alpha
    if not hasattr(alphas, '__len__'): # not a list or tuple
        alphas = [alphas] * len(ks.U_idx)

    E_U = 0.0
    logger.info(ks, "-" * 79)
    with np.printoptions(precision=5, suppress=True, linewidth=1000):
        for idx, val, lab, alpha in zip(ks.U_idx, ks.U_val, ks.U_lab, alphas):
            lab_string = " "
            for l in lab:
                lab_string += "%9s" %(l.split()[-1])
            lab_sp = lab[0].split()
            logger.info(ks, "local rdm1 of atom %s: ",
                        " ".join(lab_sp[:2]) + " " + lab_sp[2][:2])
            P = rdm1_lo[idx[:,None], idx]
            SC = np.dot(ovlp, C_ao_lo[:, idx])
            loc_sites = P.shape[-1]
            vhub_loc = (np.eye(loc_sites) - P) * (val * 0.5)
            if alpha is not None:
                # The alpha perturbation is only applied to the linear term of
                # the local density.
                E_U += alpha * P.trace()
                vhub_loc += np.eye(loc_sites) * alpha
            # vxc is a tagged array. The inplace updating avoids loosing the
            # tagged attributes.
            vxc[:] += SC.dot(vhub_loc).dot(SC.conj().T)
            E_U += (val * 0.5) * (P.trace() - np.dot(P, P).trace() * 0.5)
            logger.info(ks, "%s\n%s", lab_string, P)
            logger.info(ks, "-" * 79)

    if E_U.real < 0.0 and all(np.asarray(ks.U_val) > 0):
        logger.warn(ks, "E_U (%s) is negative...", E_U.real)
    vxc = lib.tag_array(vxc, E_U=E_U)
    return vxc

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    """
    Electronic energy for RKSpU.
    """
    if h1e is None: h1e = ks.get_hcore()
    if dm is None: dm = ks.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)

    e1 = np.einsum('ij,ji->', h1e, dm)
    tot_e = e1 + vhf.ecoul + vhf.exc + vhf.E_U
    ks.scf_summary['e1'] = e1.real
    ks.scf_summary['coul'] = vhf.ecoul.real
    ks.scf_summary['exc'] = vhf.exc.real
    ks.scf_summary['E_U'] = vhf.E_U.real
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s  EU = %s', e1, vhf.ecoul,
                 vhf.exc, vhf.E_U)
    return tot_e.real, vhf.ecoul + vhf.exc + vhf.E_U

def set_U(ks, U_idx, U_val):
    """
    Regularize the U_idx and U_val to each atom,
    and set ks.U_idx, ks.U_val, ks.U_lab.
    """
    assert len(U_idx) == len(U_val)
    ks.U_val = []
    ks.U_idx = []
    ks.U_lab = []
    mol = ks.mol

    ao_loc = mol.ao_loc_nr()
    dims = ao_loc[1:] - ao_loc[:-1]
    # atm_ids labels the atom Id for each function
    atm_ids = np.repeat(mol._bas[:,gto.ATOM_OF], dims)

    lo_labels = np.asarray(mol.ao_labels())
    for i, idx in enumerate(U_idx):
        if isinstance(idx, str):
            lab_idx = mol.search_ao_label(idx)
            # Group basis functions centered on the same atom
            for idxj in groupby(lab_idx, atm_ids[lab_idx]):
                ks.U_idx.append(idxj)
                ks.U_val.append(U_val[i])
        else:
            ks.U_idx.append(idx)
            ks.U_val.append(U_val[i])

    ks.U_val = np.asarray(ks.U_val) / HARTREE2EV

    for idx, val in zip(ks.U_idx, ks.U_val):
        ks.U_lab.append(lo_labels[idx])

    if len(ks.U_idx) == 0:
        logger.warn(ks, "No sites specified for Hubbard U. "
                    "Please check if 'U_idx' is correctly specified")

def groupby(inp, labels):
    _, where, counts = np.unique(labels, return_index=True, return_counts=True)
    return [inp[start:start+count] for start, count in zip(where, counts)]

def make_minao_lo(ks, minao_ref):
    """
    Construct minao local orbitals.
    """
    mol = ks.mol
    nao = mol.nao
    ovlp = ks.get_ovlp()
    C_ao_minao, labels = proj_ref_ao(mol, minao=minao_ref, return_labels=True)
    C_ao_minao = lo.vec_lowdin(C_ao_minao, ovlp)

    C_ao_lo = np.zeros((nao, nao))
    for idx, lab in zip(ks.U_idx, ks.U_lab):
        idx_minao = [i for i, l in enumerate(labels) if l in lab]
        assert len(idx_minao) == len(idx)
        C_ao_sub = C_ao_minao[:, idx_minao]
        C_ao_lo[:, idx] = C_ao_sub
    return C_ao_lo

def proj_ref_ao(mol, minao='minao', return_labels=False):
    """
    Get a set of reference AO spanned by the calculation basis.
    Not orthogonalized.

    Args:
        return_labels: if True, return the labels as well.
    """
    pmol = iao.reference_mol(mol, minao)
    s1 = mol.intor('int1e_ovlp', hermi=1)
    s12 = gto.intor_cross('int1e_ovlp', mol, pmol)
    s1cd = la.cho_factor(s1)
    C_ao_lo = la.cho_solve(s1cd, s12)

    if return_labels:
        labels = pmol.ao_labels()
        return C_ao_lo, labels
    else:
        return C_ao_lo

class RKSpU(rks.RKS):
    """
    DFT+U for RKS
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 'alpha'}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = lib.invalid_method('to_hf')

    def __init__(self, mol, xc='LDA,VWN',
                 U_idx=[], U_val=[], C_ao_lo='minao', minao_ref='MINAO'):
        """
        Args:
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

        Attributes:
            U_idx: same as the input.
            U_val: effectiv U-J [in AU]
            C_ao_loc: np.array
            alpha: the perturbation [in AU] used to compute U in LR-cDFT.
                Refs: Cococcioni and de Gironcoli, PRB 71, 035105 (2005)
        """
        super().__init__(mol, xc=xc)

        set_U(self, U_idx, U_val)

        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)

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
    assert isinstance(mf_plus_u, RKSpU)
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

    C_ao_lo = mf.C_ao_lo
    ovlp = mf.get_ovlp()
    C_inv = [C_ao_lo[:,local_idx].conj().T.dot(ovlp) for local_idx in mf.U_idx]

    bare_occupancies = []
    final_occupancies = []
    for alpha in alphalist:
        # All in atomic unit
        mf.alpha = alpha / HARTREE2EV
        mf.kernel(dm0=bare_dm)
        local_occ = 0
        for c in C_inv:
            C_on_site = c.dot(mf.mo_coeff)
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += rdm1_lo.trace()
        final_occupancies.append(local_occ)

        # The first iteration of SCF
        fock = mf.get_fock(dm=bare_dm)
        e, mo = mf.eig(fock, ovlp)
        local_occ = 0
        for c in C_inv:
            C_on_site = c.dot(mo)
            rdm1_lo = mf.make_rdm1(C_on_site, mf.mo_occ)
            local_occ += rdm1_lo.trace()
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
