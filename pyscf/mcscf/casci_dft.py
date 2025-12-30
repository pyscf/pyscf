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
DFT-Corrected CASCI with FOMO Support
=====================================

This module implements CASCI calculations with DFT-corrected core energy,
supporting both standard CASCI and FOMO-CASCI (Fractional Occupation 
Molecular Orbital) wavefunctions.

Theory
------
In standard CASCI, the core energy is computed using Hartree-Fock:
    E_core^HF = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J] - 0.25*Tr[D_core * K]

In DFT-corrected CASCI, the core energy uses DFT:
    E_core^DFT = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J] + E_xc[core]

The active space embedding still uses HF-like potential (J - 0.5*K) to preserve
the wavefunction topology and CI coefficients.

Examples
--------
Standard DFT-CASCI:

>>> from pyscf import gto, scf
>>> from pyscf.mcscf import casci_dft
>>> mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
>>> mf = scf.RHF(mol).run()
>>> mc = casci_dft.CASCI(mf, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()

FOMO-CASCI-DFT:

>>> from pyscf.mcscf import addons_fomo, casci_dft
>>> mf_fomo = addons_fomo.fomo_scf(mf, temperature=0.25, method='gaussian',
...                                 restricted=(ncore, ncas))
>>> mf_fomo.kernel()
>>> mc = casci_dft.CASCI(mf_fomo, ncas=6, nelecas=6, xc='PBE')
>>> mc.kernel()
>>> g = mc.Gradients().kernel()

Reference
---------
S. Pijeau and E. G. Hohenstein,
J. Chem. Theory Comput. 2017, 13, 1130-1146
https://doi.org/10.1021/acs.jctc.6b00893
"""

from functools import reduce
import numpy as np

from pyscf import lib, scf, dft
from pyscf.mcscf import casci
from pyscf.lib import logger


class CASCI(casci.CASCI):
    """
    CASCI with DFT-evaluated core energy.
    
    This class modifies the standard CASCI energy calculation to use
    DFT exchange-correlation for the core electrons instead of HF exchange.
    
    Parameters
    ----------
    mf : SCF object
        Converged RHF or FOMO-SCF mean-field object
    ncas : int
        Number of active orbitals
    nelecas : int or tuple
        Number of active electrons (or (nalpha, nbeta))
    xc : str
        Exchange-correlation functional (default: 'PBE')
    ncore : int, optional
        Number of core orbitals (default: auto-detected)
    
    Attributes
    ----------
    xc : str
        Exchange-correlation functional
    grids_level : int
        DFT integration grid level (default: 3)
    
    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from pyscf.mcscf import casci_dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = casci_dft.CASCI(mf, ncas=2, nelecas=2, xc='LDA')
    >>> mc.kernel()
    """
    
    _keys = {'xc', 'grids_level', '_grids', '_ni'}
    
    def __init__(self, mf, ncas, nelecas, xc='PBE', ncore=None):
        super().__init__(mf, ncas, nelecas, ncore)
        self.xc = xc
        self.grids_level = 3
        self._grids = None
        self._ni = None
        
    def _build_grids(self, mol=None):
        """Build DFT integration grid."""
        if mol is None:
            mol = self.mol
        if self._grids is None or self._grids.mol is not mol:
            self._grids = dft.gen_grid.Grids(mol)
            self._grids.level = self.grids_level
            self._grids.build()
        return self._grids
    
    def _get_ni(self):
        """Get numerical integrator."""
        if self._ni is None:
            self._ni = dft.numint.NumInt()
        return self._ni
    
    def edft_core(self, dm_core, mol=None):
        """
        Compute DFT energy of core density.
        
        E_core = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J_core] + E_xc[core]
        
        Parameters
        ----------
        dm_core : ndarray
            Core density matrix in AO basis
        mol : Mole, optional
            Molecule object (default: self.mol)
            
        Returns
        -------
        float
            DFT core energy
        """
        if mol is None:
            mol = self.mol
        grids = self._build_grids(mol)
        ni = self._get_ni()
        
        h1 = self.get_hcore()
        e1 = np.einsum('ij,ji->', h1, dm_core).real
        vj = scf.hf.get_jk(mol, dm_core, hermi=1, with_j=True, with_k=False)[0]
        ej = 0.5 * np.einsum('ij,ji->', vj, dm_core).real
        _, exc, _ = ni.nr_rks(mol, grids, self.xc, dm_core)
        
        return self.energy_nuc() + e1 + ej + exc
    
    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        """
        Return effective 1e Hamiltonian and DFT core energy.
        
        The effective Hamiltonian uses HF-like embedding (J - 0.5*K) to
        preserve wavefunction topology, but the core energy uses DFT.
        
        Parameters
        ----------
        mo_coeff : ndarray, optional
            MO coefficients
        ncas : int, optional
            Number of active orbitals
        ncore : int, optional
            Number of core orbitals
            
        Returns
        -------
        h1eff : ndarray
            Effective 1e Hamiltonian in active space (ncas x ncas)
        ecore : float
            DFT core energy
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore
            
        h1 = self.get_hcore()
        mo_core = mo_coeff[:, :ncore]
        mo_cas = mo_coeff[:, ncore:ncore+ncas]
        
        if ncore > 0:
            dm_core = np.dot(mo_core, mo_core.T) * 2
            # HF-like embedding for active space (preserves CI topology)
            vhf = self.get_veff(self.mol, dm_core)
            # DFT core energy
            ecore = self.edft_core(dm_core)
        else:
            vhf = 0
            ecore = self.energy_nuc()
            
        h1eff = reduce(np.dot, (mo_cas.T, h1 + vhf, mo_cas))
        return h1eff, ecore
    
    def Gradients(self):
        """Return gradient object."""
        from pyscf.grad import casci_dft as casci_dft_grad
        return casci_dft_grad.Gradients(self)


# Alias for convenience
DFTCoreCASCI = CASCI


if __name__ == '__main__':
    from pyscf import gto, scf
    
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pvdz',
        unit='Angstrom'
    )
    mf = scf.RHF(mol).run()
    
    print("=== DFT-CASCI Energy Test ===")
    mc = CASCI(mf, 6, 6, xc='PBE')
    mc.kernel()
    print(f"Energy: {mc.e_tot:.10f} Ha")
