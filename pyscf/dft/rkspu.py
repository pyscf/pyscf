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
from pyscf import __config__

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
    vxc = super(ks.__class__, ks).get_veff(
        mol, dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)

    # V_U
    C_ao_lo = ks.C_ao_lo
    ovlp = ks.get_ovlp()
    C_inv = np.dot(C_ao_lo.conj().T, ovlp)
    rdm1_lo = C_inv.dot(dm).dot(C_inv.conj().T)

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
            P = rdm1_lo[idx[:,None], idx]
            SC = np.dot(ovlp, C_ao_lo[:, idx])
            loc_sites = P.shape[-1]
            # vxc is a tagged array. The inplace updating avoids loosing the
            # tagged attributes.
            vxc[:] += SC.dot((np.eye(loc_sites) - P) * (val * 0.5)).dot(SC.conj().T)
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

    if ks.verbose >= logger.INFO:
        from pyscf.pbc.dft.krkspu import format_idx
        logger.info(ks, "-" * 79)
        logger.debug(ks, 'U indices and values: ')
        for idx, val, lab in zip(ks.U_idx, ks.U_val, ks.U_lab):
            logger.debug(ks, '%6s [%.6g eV] ==> %-100s', format_idx(idx),
                         val * HARTREE2EV, "".join(lab))
        logger.info(ks, "-" * 79)

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

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab"}

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

    def nuc_grad_method(self):
        raise NotImplementedError

def linear_response_u(mf):
    '''
    Refs:
        [1] M. Cococcioni and S. de Gironcoli, Phys. Rev. B 71, 035105 (2005)
        [2] H. J. Kulik, M. Cococcioni, D. A. Scherlis, and N. Marzari, Phys. Rev. Lett. 97, 103001 (2006)
        [3] Heather J. Kulik, J. Chem. Phys. 142, 240901 (2015)
    '''
    pass

def linear_response_u_finite_diff(mf):
    '''
    Refs:
        https://hjkgrp.mit.edu/tutorials/2011-05-31-calculating-hubbard-u/
        https://hjkgrp.mit.edu/tutorials/2011-06-28-hubbard-u-multiple-sites/
    '''
    mf = mf.copy()
    U_size = [len(x) for x in mf.U_idx]
    mf.U_val = np.repeat(mf.U_val, U_size)
    mf.U_lab = np.repeat(mf.U_lab, U_size)
    mf.U_idx = np.hstack(mf.U_idx).reshape(-1, 1)
    disp = 1e-2 # U_val are stored in a.u.
    resp = np.empty_like(mf.U_val)
    for i in mf.U_idx.ravel():
        u0 = mf.U_val[i]
        mf.U_val[i] = u0 + disp
        e1 = mf.kernel()[0]
        mf.U_val[i] = u0 - disp
        e2 = mf.kernel()[0]
        mf.U_val[i] = u0
        resp[i] = (e1 - e2) / (disp*2)
    return resp
