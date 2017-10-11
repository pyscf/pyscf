#!/usr/bin/env python

'''
Force the FCI solver of CASSCF solving particular spin state.
'''

from pyscf import scf
from pyscf import gto
from pyscf import mcscf

mol = gto.M(
    atom = '''
      8  0  0  0
      8  0  0  1.1''',
    basis = 'ccpvdz',
    symmetry = True,
    spin = 2,
)

mf = scf.RHF(mol)
mf.kernel()

# Specify CI wfn spatial symmetry by assigning fcisolver.wfnsym
mc = mcscf.CASSCF(mf, 8, 12)
mc.fcisolver.wfnsym = 'A2u'
mc.kernel()
print('Triplet Sigma_u^- %.15g  ref = -149.383495797891' % mc.e_tot)

# Specify CI wfn spatial symmetry and spin symmetry
mc.fix_spin_(ss=6)  # Quintet, ss = S*(S+1) = 6
mc.fcisolver.wfnsym = 'A2u'
mc.kernel()
print('Quintet Sigma_u^- %.15g  ref = -148.920732172378' % mc.e_tot)


#
# Similarly, you can get ground state wfn of triplet Sz=0
#
mc = mcscf.CASSCF(mf, 8, 12)#(6,6))
#mc.fcisolver = fci.direct_spin1_symm.FCI(mol)
#fci.addons.fix_spin_(mc.fcisolver, ss=2)
#mc.fcisolver.wfnsym = 'A2g'
mc.kernel()
print('Triplet Sigma_g^- %.15g  ref = -149.688656224059' % mc.e_tot)


#
# In the following example, without fix_spin_ decoration, it's probably unable
# to converge to the correct spin state.
#
mol = gto.M(
    atom = 'Mn 0 0 0; Mn 0 0 2.5',
    basis = 'ccpvdz',
    symmetry = 1,
)
mf = scf.RHF(mol)
mf.set(level_shift=0.4).run()
mc = mcscf.CASCI(mf, 12, 14)
mc.fcisolver.max_cycle = 100
mo = mc.sort_mo_by_irrep({'A1g': 2, 'A1u': 2,
                          'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1,
                          'E2uy': 1, 'E2ux': 1, 'E2gy': 1, 'E2gx': 1},
                         {'A1g': 5, 'A1u': 5,
                          'E1uy': 2, 'E1ux': 2, 'E1gy': 2, 'E1gx': 2})
mc.kernel(mo)

mc.fix_spin_(shift=.5, ss=0)
mc.kernel(mo)

