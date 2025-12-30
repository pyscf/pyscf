#!/usr/bin/env python
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
# Author: Arshad Mehmood, IACS, Stony Brook University 
# Email: arshad.mehmood@stonybrook.edu
# Date: 30 December 2025

"""
Nuclear Gradients for DFT-Corrected CASCI with FOMO Support
============================================================

This module implements nuclear gradients for CASCI calculations with 
DFT-corrected core energy, supporting both standard CASCI and FOMO-CASCI
(Fractional Occupation Molecular Orbital) wavefunctions.

The gradient is computed using numerical differentiation to ensure
correctness for both standard and FOMO-CASCI cases.

Examples
--------
>>> from pyscf import gto, scf
>>> from pyscf.mcscf import casci_dft
>>> mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
>>> mf = scf.RHF(mol).run()
>>> mc = casci_dft.CASCI(mf, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()
>>> g = mc.Gradients().kernel()
"""

from functools import reduce
import numpy as np

from pyscf import gto, lib, scf, ao2mo
from pyscf.lib import logger
from pyscf.grad import casci as casci_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange


def grad_elec_fomo(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    """
    CASCI electronic gradient with FOMO support.
    
    This is based on PySCF's casci.grad_elec but modified to handle
    FOMO's fractional occupations by skipping the CPHF orbital response.
    
    For FOMO-CASCI, the orbitals come from a FOMO-SCF calculation and are
    not variationally optimized for the CASCI energy. The orbital response
    contribution is therefore neglected (set to zero).
    
    Parameters
    ----------
    mc_grad : Gradients object
        Gradient calculator
    mo_coeff : ndarray, optional
        MO coefficients
    ci : ndarray, optional
        CI coefficients
    atmlst : list, optional
        List of atom indices for gradient calculation
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    de : ndarray
        Electronic gradient (natm x 3)
    """
    mc = mc_grad.base
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao + 1) // 2

    mo_occ = mo_coeff[:, :nocc]
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    # Build density matrices
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    
    # Two-electron integrals
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
    aapa = aapa.reshape(ncas, ncas, nmo, ncas)
    
    # Fock-like matrices
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * 0.5
    vhf_a = vj[1] - vk[1] * 0.5
    
    # Lagrangian matrix
    Imat = np.zeros((nmo, nmo))
    Imat[:, :nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
    Imat[:, ncore:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    Imat[:, ncore:nocc] += lib.einsum('uviw,vuwt->it', aapa, casdm2)
    aapa = vj = vk = vhf_c = vhf_a = h1 = None

    # For FOMO: Skip CPHF, set orbital response to zero
    log.debug("FOMO-CASCI: Skipping CPHF (orbital response set to zero)")
    im1 = reduce(np.dot, (mo_coeff, Imat, mo_coeff.T))

    casci_dm1 = dm_core + dm_cas
    hcore_deriv = mc_grad.hcore_generator(mol)
    s1 = mc_grad.get_ovlp(mol)

    # Transform 2-RDM to AO basis
    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2, ncas**2), mo_cas.T,
                                 (0, nao, 0, nao)).reshape(ncas**2, nao, nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:, diag_idx] *= 0.5
    dm2buf = dm2buf.reshape(ncas, ncas, nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = list(range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de = np.zeros((len(atmlst), 3))

    max_memory = mc_grad.max_memory - lib.current_memory()[0]
    blksize = int(max_memory * 0.9e6 / 8 / ((aoslices[:, 3] - aoslices[:, 2]).max() * nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += np.einsum('xij,ij->x', h1ao, casci_dm1)

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3, p1-p0, nf, nao_pair)
            de[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf, -1))
                eri1tmp = eri1tmp.reshape(p1-p0, nf, nao, nao)
                
                # Coulomb contributions
                de[k, i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_core, casci_dm1[p0:p1, q0:q1]) * 2
                de[k, i] -= np.einsum('ijkl,lk,ij', eri1tmp, dm_cas, dm_core[p0:p1, q0:q1]) * 2
                
                # Exchange contributions (HF embedding)
                de[k, i] += np.einsum('ijkl,jk,il', eri1tmp, dm_core[q0:q1], casci_dm1[p0:p1])
                de[k, i] += np.einsum('ijkl,jk,il', eri1tmp, dm_cas[q0:q1], dm_core[p0:p1])
            eri1 = eri1tmp = None

        # Overlap derivative contribution
        de[k] -= np.einsum('xij,ij->x', s1[:, p0:p1], im1[p0:p1])
        de[k] -= np.einsum('xij,ji->x', s1[:, p0:p1], im1[:, p0:p1])

    log.timer('CASCI nuclear gradients', *time0)
    return de


def kernel(mc_grad, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
    """
    Compute DFT-CASCI nuclear gradient.
    
    Parameters
    ----------
    mc_grad : Gradients object
        Gradient calculator
    mo_coeff : ndarray, optional
        MO coefficients
    ci : ndarray, optional
        CI coefficients
    atmlst : list, optional
        List of atom indices for gradient calculation
    state : int, optional
        State index for state-averaged CASCI
    verbose : int, optional
        Verbosity level
        
    Returns
    -------
    de : ndarray
        Nuclear gradient (natm x 3)
    """
    from pyscf.mcscf import casci_dft, addons_fomo
    
    log = logger.new_logger(mc_grad, verbose)
    mc = mc_grad.base
    mol = mc.mol
    
    if atmlst is None:
        atmlst = list(range(mol.natm))
    if state is not None:
        mc_grad.state = state
    
    ncas = mc.ncas
    nelecas = mc.nelecas
    ncore = mc.ncore
    xc = mc.xc
    step = mc_grad.numerical_step
    
    # Check if FOMO
    scf_mo_occ = mc._scf.mo_occ
    is_fomo = np.any((scf_mo_occ > 0.01) & (scf_mo_occ < 1.99))
    
    log.info("Computing numerical gradient (step=%.1e Bohr)...", step)
    
    # Get FOMO parameters if applicable
    if is_fomo:
        fomo_temp = getattr(mc._scf, 'fomo_temperature', 0.25)
        fomo_method = getattr(mc._scf, 'fomo_method', 'gaussian')
    
    atoms = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = [list(mol.atom_coord(i)) for i in range(mol.natm)]
    
    de = np.zeros((len(atmlst), 3))
    
    for k, ia in enumerate(atmlst):
        for icoord in range(3):
            # Forward displacement
            coords_p = [list(c) for c in coords]
            coords_p[ia][icoord] += step
            
            # Backward displacement
            coords_m = [list(c) for c in coords]
            coords_m[ia][icoord] -= step
            
            # Build displaced molecules
            atom_str_p = '; '.join([f'{atoms[i]} {coords_p[i][0]} {coords_p[i][1]} {coords_p[i][2]}' 
                                    for i in range(mol.natm)])
            atom_str_m = '; '.join([f'{atoms[i]} {coords_m[i][0]} {coords_m[i][1]} {coords_m[i][2]}' 
                                    for i in range(mol.natm)])
            
            mol_p = gto.M(atom=atom_str_p, basis=mol.basis, unit='Bohr', verbose=0)
            mol_m = gto.M(atom=atom_str_m, basis=mol.basis, unit='Bohr', verbose=0)
            
            # Run SCF
            mf_p = scf.RHF(mol_p).run()
            mf_m = scf.RHF(mol_m).run()
            
            if is_fomo:
                mf_p = addons_fomo.fomo_scf(mf_p, temperature=fomo_temp, 
                                            method=fomo_method, restricted=(ncore, ncas))
                mf_p.kernel()
                mf_m = addons_fomo.fomo_scf(mf_m, temperature=fomo_temp,
                                            method=fomo_method, restricted=(ncore, ncas))
                mf_m.kernel()
            
            # Run DFT-CASCI
            mc_p = casci_dft.CASCI(mf_p, ncas, nelecas, xc=xc)
            mc_p.kernel(verbose=0)
            mc_m = casci_dft.CASCI(mf_m, ncas, nelecas, xc=xc)
            mc_m.kernel(verbose=0)
            
            # Central difference
            de[k, icoord] = (mc_p.e_tot - mc_m.e_tot) / (2 * step)
    
    return de


class Gradients(casci_grad.Gradients):
    """
    Gradients for DFT-corrected CASCI with FOMO support.
    
    This class computes nuclear gradients for DFT-CASCI calculations.
    The gradient is computed using numerical differentiation to ensure
    correctness for both standard and FOMO-CASCI cases.
    
    Attributes
    ----------
    numerical_step : float
        Step size for numerical differentiation (default: 1e-4 Bohr)
    
    Examples
    --------
    >>> from pyscf.mcscf import casci_dft
    >>> mc = casci_dft.CASCI(mf, ncas, nelecas, xc='PBE')
    >>> mc.kernel()
    >>> g = mc.Gradients()
    >>> de = g.kernel()
    """
    
    def __init__(self, mc):
        super().__init__(mc)
        self.numerical_step = 1e-4
    
    def grad_elec(self, mo_coeff=None, ci=None, atmlst=None, verbose=None):
        """
        Compute electronic gradient.
        
        Automatically detects FOMO via fractional occupations.
        """
        mc = self.base
        scf_mo_occ = mc._scf.mo_occ
        is_fomo = np.any((scf_mo_occ > 0.01) & (scf_mo_occ < 1.99))
        
        if is_fomo:
            return grad_elec_fomo(self, mo_coeff, ci, atmlst, verbose)
        else:
            return casci_grad.grad_elec(self, mo_coeff, ci, atmlst, verbose)
    
    def kernel(self, mo_coeff=None, ci=None, atmlst=None, state=None, verbose=None):
        """
        Compute DFT-CASCI nuclear gradient.
        
        Parameters
        ----------
        mo_coeff : ndarray, optional
            MO coefficients
        ci : ndarray, optional
            CI coefficients
        atmlst : list, optional
            List of atom indices for gradient calculation
        state : int, optional
            State index for state-averaged CASCI
        verbose : int, optional
            Verbosity level
            
        Returns
        -------
        de : ndarray
            Nuclear gradient (natm x 3)
        """
        log = logger.new_logger(self, verbose)
        mc = self.base
        mol = mc.mol
        
        if atmlst is None:
            atmlst = list(range(mol.natm))
        if state is not None:
            self.state = state
        
        # Compute numerical gradient
        de = kernel(self, mo_coeff, ci, atmlst, state, verbose)
        
        self.de = de
        
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
            
        log.note('--------- %s gradients for state %d ----------',
                 mc.__class__.__name__, self.state)
        rhf_grad._write(log, mol, self.de, atmlst)
        log.note('----------------------------------------------')
        
        return self.de


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.mcscf import casci_dft
    
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pvdz',
        unit='Angstrom'
    )
    mf = scf.RHF(mol).run()
    
    print("=== DFT-CASCI Gradient Test ===")
    mc = casci_dft.CASCI(mf, 6, 6, xc='PBE')
    mc.kernel()
    print(f"Energy: {mc.e_tot:.10f} Ha")
    g = mc.Gradients().kernel()
