#!/usr/bin/env/python

from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

# 1. CASCI density

mc0 = mcpdft.CASCI (mf, 'tPBE', 6, 8).run ()

# 2. CASSCF density
# Note that the MC-PDFT energy may not be lower, even though
# E(CASSCF)<=E(CASCI).

mc1 = mcpdft.CASSCF (mf, 'tPBE', 6, 8).run ()

# 3. analyze () does the same thing as CASSCF analyze ()

mc1.verbose = 4
mc1.analyze ()

# 4. Energy decomposition for additional analysis

e_decomp = mc1.get_energy_decomposition (split_x_c=False)
print ("e_nuc =",e_decomp[0])
print ("e_1e =",e_decomp[1])
print ("e_Coul =",e_decomp[2])
print ("e_OT =",e_decomp[3])
print ("e_ncwfn (not included in total energy) =",e_decomp[4])
print ("e_PDFT - e_MCSCF =", mc1.e_tot - mc1.e_mcscf)
print ("e_OT - e_ncwfn =", e_decomp[3] - e_decomp[4])
e_decomp = mc1.get_energy_decomposition (split_x_c=True)
print ("e_OT (x component) = ",e_decomp[3])
print ("e_OT (c component) = ",e_decomp[4])


