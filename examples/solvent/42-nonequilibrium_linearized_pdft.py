#!/usr/bin/env python
#
# Author: Arpan Choudhury (University of Chicago) <arpanchoudhury29@gmail.com>
#

'''
Nonequilibrium solvation: partitions the solvent charges into
slow (from the ground state) and fast (from an excited state) components.
Useful for vertical excitation calculations.
'''

from pyscf import gto, scf, mcpdft
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


mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4, grids_level=6)
mc.fix_spin_(ss=0)
mc_lpdft =  mc.multi_state([0.5, 0.5], "lin")

mc_lpdft = solvent.PCM(mc_lpdft)
mc_lpdft.with_solvent.equilibrium_solvation = False
mc_lpdft.with_solvent.rfroot = 1 # fast charges from first excited state
mc_lpdft.kernel()
