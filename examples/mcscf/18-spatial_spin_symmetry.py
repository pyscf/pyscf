#!/usr/bin/env python

from pyscf import scf
from pyscf import gto
from pyscf import mcscf, fci
from pyscf.mcscf import dmet_cas

'''
Force the FCI solver of CASSCF following certain spin state.
'''

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
fci.addons.fix_spin_(mc.fcisolver, ss_value=6)  # Quintet, ss_value = S*(S+1) = 6
mc.fcisolver.wfnsym = 'A2u'
mc.kernel()
print('Quintet Sigma_u^- %.15g  ref = -148.920732172378' % mc.e_tot)


#
# Similarly, you can get ground state wfn of triplet Sz=0
#
mc = mcscf.CASSCF(mf, 8, 12)#(6,6))
#mc.fcisolver = fci.direct_spin1_symm.FCISolver(mol)
#fci.addons.fix_spin_(mc.fcisolver, ss_value=2)
#mc.fcisolver.wfnsym = 'A2g'
mc.kernel()
print('Triplet Sigma_g^- %.15g  ref = -149.688656224059' % mc.e_tot)
