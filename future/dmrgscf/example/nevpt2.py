#!/usr/bin/env python
#
# Author: Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import mrpt
from pyscf import dmrgscf

'''
DMRG-CASCI then DMRG-NEVPT2 calculation.

There are two NEVPT2 implementations available for DMRG Block program.  The slow
version (default) strictly follows the formula presented in JCP, 117(2002), 9138
in which the 4-particle density matrix is explictly computed.  Typically 26
orbitals is the upper limit of the slow version due to the large requirements
on the memory usage.  The fast version employs the so called MPS-pertuber
technique.  It is able to handle much larger systems, up to about 30 orbitals.
'''

#
# One can adjust the processor numbers for Block code on the runtime.
#
#from pyscf.dmrgscf import settings
#settings.MPIPREFIX ='mpirun -n 3'

b = 1.4
mol = gto.Mole()
mol.build(
    verbose = 4,
    output = 'fci_nevpt2.out',
    atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
    basis = 'sto-3g',
#
# Note if symmetry is specified, the molecule has to be placed on the proper
# orientation to match the given symmetry.
#
    symmetry = 'd2h',
)
m = scf.RHF(mol)
m.kernel()

#
# FCI-based CASCI + NEVPT2. Two roots are computed.  So CASCI attribute mc.ci
# holds the two CI wave functions.  They need to be passed to sc_nevpt
# function to control the state-specific NEVPT2 calculation.  By default the
# lowest root will be computed.
#
mc = mcscf.CASCI(m, 4, 4)
mc.fcisolver.nroots = 2
mc.casci()

ci_nevpt_e1 = mrpt.sc_nevpt(mc, ci=mc.ci[0])
ci_nevpt_e2 = mrpt.sc_nevpt(mc, ci=mc.ci[1])

#
# By default, the orbitals are canonicalized after calling CASCI solver.  Save
# the canonical MC orbitals for later use.  Althoug it's not necessary for this
# example, we put here to demonstrate how to carry out CASCI calculation on
# given orbitals (or as initial guess for CASSCF method) other than the
# default HF canonical orbitals.  More examples of initial guess refers to
# pyscf/mcscf/33-make_init_guess.
#
mc_orb = mc.mo_coeff

##################################################
#
# DMRG-NEVPT2 slow version
# 4-particle density matrix is explicitly computed.
# Ref: S Guo, JCTC, ASAP
#
##################################################

#
# Call mol.build function to redirect the output to another file.
#
mol.build(output = 'dmrg_nevpt2_slow.out')

mc = mcscf.CASCI(m, 4, 4)
#
# Use DMRGCI as the active space solver.  DMRGCI solver allows us to handle
# ~ 50 active orbitals.
#
mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=200)
mc.fcisolver.nroots = 2
#
# Passing mc_orb to CASCI kernel function so that the CASCI calculation is
# carried out with the given orbitals.  More examples refers to
# pyscf/mcscf/62-make_init_guess
#
mc.kernel(mc_orb)

#
# In current pyscf release, the default sc_nevpt function leads to the
# DMRG-SC-NEVPT2 implementation based on the 4-particle density matrix.
#
dmrg_nevpt_e1 = mrpt.sc_nevpt(mc,ci=mc.ci[0])
dmrg_nevpt_e2 = mrpt.sc_nevpt(mc,ci=mc.ci[1])



##################################################
#
# DMRG-NEVPT2 fast version
# Use compressed MPS as perturber functions for SC-NEVPT2.
# 4-particle density matrix is not computed.
#
##################################################

#
# Use compress_perturb function to initialize compressed perturber.
# root=0 indicates that it's the perturber for ground state.
#
mps_nevpt_e1 = mrpt.sc_nevpt(dmrgscf.compress_perturb(mc, maxM=100, root=0))

#
# root=1 for first excited state.
#
mps_nevpt_e1 = mrpt.sc_nevpt(dmrgscf.compress_perturb(mc, maxM=100, root=1))


print('CI NEVPT = %.15g %.15g  DMRG NEVPT = %.15g %.15g  MPS NEVPT = %.15g %.15g'
      % (ci_nevpt_e1, ci_nevpt_e2,
         dmrg_nevpt_e1, dmrg_nevpt_e2, mps_nevpt_e1, mps_nevpt_e2,))

