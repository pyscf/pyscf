#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.dmrgscf.dmrgci import DMRGCI, DMRGSCF

'''
When Block code is used for active space solver, state-average calculation can
be set up in various ways.
'''

b = 1.2
mol = gto.M(
    verbose = 4,
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

#
# Block code has built-in state-average feature.  To active the state-average
# calculation via Block code, one needs set the nroots and weights attributes
# of DMRGCI class.
#
dmrgsolver = DMRGCI(mol)
weights = [0.5, 0.5]
dmrgsolver.weights = weights
dmrgsolver.nroots  = len(weights)
mc = mcscf.CASSCF(m, 8, 8)
mc.fcisolver = dmrgsolver
mc.kernel()
print(mc.e_tot)


#
# The previous settings can be simplified with the wrapper DMRGSCF.
# DMRGSCF provided a simple way to specify the state-average calculation.
#
mc = DMRGSCF(m, 8, 8)
mc.state_average_([0.5, 0.5])
mc.kernel()
print(mc.e_tot)


#
# More general and/or complicated state-average calculations:
#
# Block code does not allow state average over different spin symmetry or
# spatial symmetry.  Mixing different spin or different irreducible
# representations requires multiple passes of DMRG calculations.
# See also pyscf/examples/mcscf/41-hack_state_average.py
#

#
# state-average over different spin states
#
weights = [.25, .25, .5]  # 0.25 singlet + 0.25 singlet + 0.5 triplet
dmrgsolver1 = DMRGCI(mol)
dmrgsolver1.nroots = 2
dmrgsolver1.weights = [.5, .5]
dmrgsolver1.scratchDirectory = '/scratch/dmrg1'
dmrgsolver2 = DMRGCI(mol)
dmrgsolver2.scratchDirectory = '/scratch/dmrg2'

class FakeCISolver(DMRGCI):
    def kernel(self, h1, h2, norb, nelec, *args, **kwargs):
        # singlet
        e1, r1 = dmrgsolver1.kernel(h1, h2, norb, (nelec//2,nelec//2))
        # triplet
        e2, r2 = dmrgsolver2.kernel(h1, h2, norb, (nelec//2+1,nelec//2-1))
        e_avg = e1[0]*(weights[0]+weights[1]) + e2*weights[2]
        return e_avg, [r1[0], r1[1], r2]

    def approx_kernel(self, h1, h2, norb, nelec, *args, **kwargs):
        # singlet
        e1, r1 = dmrgsolver1.approx_kernel(h1, h2, norb, (nelec//2,nelec//2))
        # triplet
        e2, r2 = dmrgsolver2.approx_kernel(h1, h2, norb, (nelec//2+1,nelec//2-1))
        e_avg = e1[0]*(weights[0]+weights[1]) + e2*weights[2]
        return e_avg, [r1[0], r1[1], r2]

    def make_rdm1(self, state, norb, nelec):
        dm1_1 = dmrgsolver1.make_rdm1(state[0], norb, (nelec//2,nelec//2))
        dm1_2 = dmrgsolver1.make_rdm1(state[1], norb, (nelec//2,nelec//2))
        dm1_3 = dmrgsolver2.make_rdm1(state[2], norb, (nelec//2+1,nelec//2-1))
        rdm1 = dm1_1 * weights[0] + dm1_2 * weights[1] + dm1_3 * weights[2]
        return rdm1

    def make_rdm12(self, state, norb, nelec):
        dm12_1 = dmrgsolver1.make_rdm12(state[0], norb, (nelec//2,nelec//2))
        dm12_2 = dmrgsolver1.make_rdm12(state[1], norb, (nelec//2,nelec//2))
        dm12_3 = dmrgsolver2.make_rdm12(state[2], norb, (nelec//2+1,nelec//2-1))
        rdm1 = dm12_1[0] * weights[0] + dm12_2[0] * weights[1] + dm12_3[0] * weights[2]
        rdm2 = dm12_1[1] * weights[0] + dm12_2[1] * weights[1] + dm12_3[1] * weights[2]
        return rdm1, rdm2

mc = mcscf.CASSCF(m, 8, 8)
mc.fcisolver = FakeCISolver(mol)
mc.kernel()
print(mc.e_tot)




#
# state-average over states of different spatial symmetry
#
weights = [.2, .4, .4]
dmrgsolver1 = DMRGCI(mol)
dmrgsolver1.wfnsym = 'Ag'
dmrgsolver1.scratchDirectory = '/scratch/dmrg1'
dmrgsolver2 = DMRGCI(mol)
dmrgsolver2.wfnsym = 'B1g'
dmrgsolver2.scratchDirectory = '/scratch/dmrg2'
dmrgsolver3 = DMRGCI(mol)
dmrgsolver3.wfnsym = 'B1u'
dmrgsolver3.scratchDirectory = '/scratch/dmrg3'

class FakeCISolver1(DMRGCI):
    def kernel(self, h1, h2, norb, nelec, *args, **kwargs):
        e1, r1 = dmrgsolver1.kernel(h1, h2, norb, nelec)
        e2, r2 = dmrgsolver2.kernel(h1, h2, norb, nelec)
        e3, r3 = dmrgsolver3.kernel(h1, h2, norb, nelec)
        e_avg = e1*weights[0] + e2*weights[1] + e3*weights[2]
        return e_avg, [r1, r2, r3]

    def approx_kernel(self, h1, h2, norb, nelec, *args, **kwargs):
        e1, r1 = dmrgsolver1.approx_kernel(h1, h2, norb, nelec)
        e2, r2 = dmrgsolver2.approx_kernel(h1, h2, norb, nelec)
        e3, r3 = dmrgsolver3.approx_kernel(h1, h2, norb, nelec)
        e_avg = e1*weights[0] + e2*weights[1] + e3*weights[2]
        return e_avg, [r1, r2, r3]

    def make_rdm1(self, state, norb, nelec):
        dm1_1 = dmrgsolver1.make_rdm1(state[0], norb, nelec)
        dm1_2 = dmrgsolver2.make_rdm1(state[1], norb, nelec)
        dm1_3 = dmrgsolver3.make_rdm1(state[2], norb, nelec)
        rdm1 = dm1_1 * weights[0] + dm1_2 * weights[1] + dm1_3 * weights[2]
        return rdm1

    def make_rdm12(self, state, norb, nelec):
        dm12_1 = dmrgsolver1.make_rdm12(state[0], norb, nelec)
        dm12_2 = dmrgsolver2.make_rdm12(state[1], norb, nelec)
        dm12_3 = dmrgsolver3.make_rdm12(state[2], norb, nelec)
        rdm1 = dm12_1[0] * weights[0] + dm12_2[0] * weights[1] + dm12_3[0] * weights[2]
        rdm2 = dm12_1[1] * weights[0] + dm12_2[1] * weights[1] + dm12_3[1] * weights[2]
        return rdm1, rdm2

mc = mcscf.CASSCF(m, 8, 8)
mc.fcisolver = FakeCISolver1(mol)
mc.kernel()
print(mc.e_tot)
