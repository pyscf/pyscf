#!/usr/bin/env python

'''
ppRPA excitation energies in the hole-hole channel
'''

from pyscf import gto, dft

# For ppRPA excitation energy of N-electron system in hole-hole channel
# mean field is (N+2)-electron

mol = gto.Mole()
mol.verbose = 5
mol.atom = [
    ["O", (0.0, 0.0, 0.0)],
    ["H", (0.0, -0.7571, 0.5861)],
    ["H", (0.0, 0.7571, 0.5861)]]
mol.basis = 'def2-svp'
# create a (N+2)-electron system for charged-neutral H2O
mol.charge = -2
mol.build()

# =====> Part I. Restricted ppRPA <=====
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()

# direct diagonalization, N6 scaling
# ppRPA can be solved in an active space
from pyscf.pprpa.rpprpa_direct import RppRPADirect
pp = RppRPADirect(mf, nvir_act=10, nelec="n+2")
# number of two-electron removal states to print
pp.hh_state = 10
# solve for singlet states
pp.kernel("s")
# solve for triplet states
pp.kernel("t")
pp.analyze()

# Davidson algorithm, N4 scaling
# ppRPA can be solved in an active space
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson
pp = RppRPADavidson(mf, nvir_act=10, channel="hh")
# solve for singlet states
pp.kernel("s")
# solve for triplet states
pp.kernel("t")
pp.analyze()

# =====> Part II. Unrestricted ppRPA <=====
# unrestricted KS-DFT calculation as starting point of UppRPA
umf = dft.UKS(mol)
umf.xc = "b3lyp"
umf.kernel()

# direct diagonalization, N6 scaling
from pyscf.pprpa.upprpa_direct import UppRPADirect
# ppRPA can be solved in an active space
pp = UppRPADirect(umf, nvir_act=10, nelec='n+2')
# number of two-electron addition states to print
pp.pp_state = 10
# solve ppRPA in the (alpha alpha, alpha alpha) subspace
pp.kernel(subspace=['aa'])
# solve ppRPA in the (alpha beta, alpha beta) subspace
pp.kernel(subspace=['ab'])
# solve ppRPA in the (beta beta, beta beta) subspace
pp.kernel(subspace=['bb'])
pp.analyze()

