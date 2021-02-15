#!/usr/bin/env python

'''
Transition density matrix and transition dipole for CASSCF/CASCI wavefunction
'''

import numpy
from pyscf import gto, scf, mcscf, fci

mol = gto.M(atom='''
C   1.20809735,    0.69749533,   0.00000000
C   0.00000000,    1.39499067,   0.00000000
C   -1.20809735,   0.69749533,   0.00000000
C   -1.20809735,  -0.69749533,   0.00000000
C   -0.00000000,  -1.39499067,   0.00000000
C   1.20809735,   -0.69749533,   0.00000000
H   2.16038781,    1.24730049,   0.00000000
H   0.00000000,    2.49460097,   0.00000000
H   -2.16038781,   1.24730049,   0.00000000
H   -2.16038781,  -1.24730049,   0.00000000
H   0.00000000,   -2.49460097,   0.00000000
H   2.16038781,   -1.24730049,   0.00000000''',
verbose = 4,
basis = 'ccpvdz',
symmetry = True,
)
mf = scf.RHF(mol)
mf.kernel()
#mf.analyze()

#
# 1. State-average CASSCF to get optimal orbitals
#
mc = mcscf.CASSCF(mf, 6, 6)
solver_ag = fci.direct_spin0_symm.FCI(mol)
solver_b2u = fci.direct_spin0_symm.FCI(mol)
solver_b2u.wfnsym = 'B2u'
mc.fcisolver = mcscf.state_average_mix(mc, [solver_ag,solver_b2u], [.5,.5])
cas_list = [17,20,21,22,23,30]  # 2pz orbitals
mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list)
mc.kernel(mo)
#mc.analyze()
mc_mo = mc.mo_coeff

#
# 2. Ground state wavefunction.  This step can be passed you approximate it
# with the state-averaged CASSCF wavefunction
#
mc = mcscf.CASCI(mf, 6, 6)
mc.fcisolver.wfnsym = 'Ag'
mc.kernel(mc_mo)
ground_state = mc.ci

#
# 3. Exited states.  In this example, B2u are bright states.
#
# Here, mc.ci[0] is the first excited state.
#
mc = mcscf.CASCI(mf, 6, 6)
mc.fcisolver.wfnsym = 'B2u'
mc.fcisolver.nroots = 8
mc.kernel(mc_mo)

#
# 4. transition density matrix and transition dipole
#
# Be careful with the gauge origin of the dipole integrals
#
charges = mol.atom_charges()
coords = mol.atom_coords()
nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
mol.set_common_orig_(nuc_charge_center)
dip_ints = mol.intor('cint1e_r_sph', comp=3)

def makedip(ci_id):
    # transform density matrix in MO representation
    t_dm1 = mc.fcisolver.trans_rdm1(ground_state, mc.ci[ci_id], mc.ncas, mc.nelecas)
    # transform density matrix to AO representation
    orbcas = mc_mo[:,mc.ncore:mc.ncore+mc.ncas]
    t_dm1_ao = reduce(numpy.dot, (orbcas, t_dm1, orbcas.T))
    # transition dipoles
    return numpy.einsum('xij,ji->x', dip_ints, t_dm1_ao)

# 1st and 6th excited states are B2u of D6h point group, dark states
# 3rd and 4th excited states are triplet states, dipole == 0
for i in range(8):
    print('Transition dipole between |0> and |%d>'%(i+1), makedip(i))

