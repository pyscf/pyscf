#!/usr/bin/env python

'''
ppRPA correlation energy and total energy
'''

from pyscf import gto, dft

# For ppRPA total energy of N-electron system
# mean field is also N-electron

mol = gto.Mole()
mol.verbose = 5
mol.atom = [
    ["O", (0.0, 0.0, 0.0)],
    ["H", (0.0, -0.7571, 0.5861)],
    ["H", (0.0, 0.7571, 0.5861)]]
mol.basis = 'def2-svp'
mol.build()

# =====> Part I. Restricted ppRPA <=====
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()

from pyscf.pprpa.rpprpa_direct import RppRPADirect
pp = RppRPADirect(mf)
ec = pp.get_correlation()
etot, ehf, ec = pp.energy_tot()
print("H2O Hartree-Fock energy = %.8f" % ehf)
print("H2O ppRPA correlation energy = %.8f" % ec)
print("H2O ppRPA total energy = %.8f" % etot)

# =====> Part II. Unrestricted ppRPA <=====
# unrestricted KS-DFT calculation as starting point of UppRPA
umf = dft.UKS(mol)
umf.xc = "b3lyp"
umf.kernel()

# direct diagonalization, N6 scaling
from pyscf.pprpa.upprpa_direct import UppRPADirect
pp = UppRPADirect(umf)
ec = pp.get_correlation()
etot, ehf, ec = pp.energy_tot()
print("H2O Hartree-Fock energy = %.8f" % ehf)
print("H2O ppRPA correlation energy = %.8f" % ec)
print("H2O ppRPA total energy = %.8f" % etot)

