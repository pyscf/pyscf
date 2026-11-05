#!/usr/bin/env python
#
# Author: Arpan Choudhury (University of Chicago) <arpanchoudhury29@gmail.com>
#

'''
Nonequilibrium solvation: partitions the solvent charges into
slow (from a given density matrix, usually from a priori ground state
calculation) and fast (from the excited state) components.
.
'''

from pyscf import gto, scf, mcscf
from pyscf import solvent


mol = gto.M(
    atom='''
H              1.110158    0.935182   -0.000022
C              0.537003    0.000000    0.000011
H              1.110154   -0.935184   -0.000022
O             -0.680291    0.000000   -0.000003''',
    basis='ccpvdz',
    verbose=4)

mf = solvent.PCM(scf.RHF(mol), 'water')
mf.kernel()

# === ground state equilibrium solvation
mc = mcscf.CASSCF(mf, 8, 8)
mc_eq = solvent.PCM(mc)
mc_eq.kernel()

ref_dm = mc_eq.make_rdm1()

# === state-specific excited state nonequilibrium solvation
state_id = 1 # first excited state
mc = mcscf.CASSCF(mf, 8, 8).state_specific_(state_id)
mc.fix_spin_(ss=0)
mc_neq = solvent.PCM(mc)
mc_neq.with_solvent.equilibrium_solvation = False
mc_neq.with_solvent.refdm = ref_dm
mc_neq.kernel()

