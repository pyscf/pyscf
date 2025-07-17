#!/usr/bin/env python

'''
linearized pair-density functional theory

A multi-states extension of MC-PDFT for states close in energy. Generates an
effective L-PDFT Hamiltonian through a Taylor expansion of the MC-PDFT energy
expression [J Chem Theory Comput: 10.1021/acs.jctc.3c00207].
'''

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom = [
        ['Li', ( 0., 0., 0.)],
        ['H', ( 0., 0., 3)]
    ], basis = 'sto-3g',
    symmetry = 0
    )

mf = scf.RHF(mol)
mf.kernel()

mc = mcpdft.CASSCF(mf, 'tpbe', 2, 2)
mc.fix_spin_(ss=0) # often necessary!
mc_sa = mc.state_average ([.5, .5]).run ()
# Note, this operates in the same way as "CMS-PDFT" but constructs and
# diagonalizes a different matrix.
mc_ms = mc.multi_state ([.5, .5], "lin").run (verbose=4)

print ('{:>21s} {:>12s} {:>12s}'.format ('state 0','state 1', 'gap'))
fmt_str = '{:>9s} {:12.9f} {:12.9f} {:12.9f}'
print (fmt_str.format ('CASSCF', mc_sa.e_mcscf[0], mc_sa.e_mcscf[1],
    mc_sa.e_mcscf[1]-mc_sa.e_mcscf[0]))
print (fmt_str.format ('MC-PDFT', mc_sa.e_states[0], mc_sa.e_states[1],
    mc_sa.e_states[1]-mc_sa.e_states[0]))
print (fmt_str.format ('L-PDFT', mc_ms.e_states[0], mc_ms.e_states[1],
    mc_ms.e_states[1]-mc_ms.e_states[0]))
