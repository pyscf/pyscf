#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CIS excited state density with TDA amplitudes
'''

import numpy as np
from pyscf import gto, dft, tdscf

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = '631g',
)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mytd = tdscf.TDA(mf).run(nstates=3)
#mytd.analyze()

def tda_denisty_matrix(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

# Density matrix for the 3rd excited state
dm = tda_denisty_matrix(mytd, 2)

# Write to cube format
from pyscf.tools import cubegen
cubegen.density(mol, 'tda_density.cube', dm)

# Write the difference between excited state and ground state
cubegen.density(mol, 'density_diff.cube', dm-mf.make_rdm1())

# The positive and negative parts can be overlayed in Jmol
# isosurface ID "surf1" cutoff  0.02 density_diff.cube
# isosurface ID "surf2" cutoff -0.02 density_diff.cube
