#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
State average

Using mcscf.state_average_ to decorate the CASSCF object
'''

from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g',
    symmetry = 1)

mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.state_average_(mcscf.CASSCF(mf, 4, 4), [.64,.36])
mc.verbose = 4
mc.kernel()
mo = mc.mo_coeff

# An equivalent input for state average is
mc = mcscf.CASSCF(mf, 4, 4).state_average_([.64,.36]).run(verbose=4)
#
# Note stream operations are applied here.  The above one line code is
# equivalent to the following serail statements
#
#mc = mcscf.CASSCF(mf, 4, 4)
#mc.state_average_([.64,.36])
#mc.verbose = 4
#mc.kernel()
mo = mc.mo_coeff

#
# Extra CASCI for ground state energy because the state_averaged CASSCF
# computes the state-averaged total energy instead of the ground state energy.
#
mc = mcscf.CASCI(mf, 4, 4)
emc = mc.casci(mo)[0]

print('E(CAS) = %.12f, ref = -75.982521066893' % emc)


#
# Universal spin multiplicity can be assigned to the state-averaged FCI
# solver.  If some states break the required spin, e.g. in the following
# example, a quintet may appear in the solution, the mcscf.fix_spin_ method
# can be called to force the spin symmetry (see also the example
# 18-spatial_spin_symmetry.py).
#
mc = mcscf.CASSCF(mf, 6, 8)
mc = mc.state_average_([.25, .25, .25, .25])
mc.fcisolver.spin = 2
mc.fix_spin_(ss=2)
mc.kernel()

