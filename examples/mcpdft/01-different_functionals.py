#!/usr/bin/env/python

from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

mc = mcpdft.CASCI (mf, 'tPBE', 6, 8).run ()

# 1. Change the functional by setting mc.otxc

mc.otxc = 'ftBLYP'
mc.kernel ()

# 2. Change the functional and compute energy in one line without
# reoptimizing the wave function using compute_pdft_energy_

mc.compute_pdft_energy_(otxc='ftPBE')

#
# The leading "t" or "ft" identifies either a "translated functional"
# [JCTC 10, 3669 (2014)] or a "fully-translated functional"
# [JCTC 11, 4077 (2015)]. It can be combined with a general PySCF
# xc string containing any number of pure LDA or GGA functionals.
# Meta-GGAs, built-in hybrid functionals ("B3LYP"), and range-separated
# functionals are not supported.
# 

# 3. A translated user-defined compound functional

mc.compute_pdft_energy_(otxc="t0.3*B88 + 0.7*SLATER,0.4*VWN5+0.6*LYP")

# 4. A fully-translated functional consisting of "exchange" only

mc.compute_pdft_energy_(otxc="ftPBE,")

# 5. A fully-translated functional consisting of "correlation" only

mc.compute_pdft_energy_(otxc="ft,PBE")

# 6. The sum of 5 and 6 (look at the "Eot" output)

mc.compute_pdft_energy_(otxc="ftPBE")


