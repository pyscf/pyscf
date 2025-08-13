#!/usr/bin/env python

'''
State average

This works the same way as mcscf.state_average_, but you
must use the method attribute (mc.state_average, mc.state_average_)
instead of the function call.
'''

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g',
    symmetry = 1)

mf = scf.RHF(mol)
mf.kernel()

mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4).state_average_([.64,.36]).run (verbose=4)

print ("Average MC-PDFT energy =", mc.e_tot)
print ("E_PDFT-E_CASSCF for state 0 =", mc.e_states[0]-mc.e_mcscf[0])
print ("E_PDFT-E_CASSCF for state 1 =", mc.e_states[1]-mc.e_mcscf[1])


