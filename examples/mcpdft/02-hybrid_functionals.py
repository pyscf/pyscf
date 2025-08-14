#!/usr/bin/env/python


#
# JPCL 11, 10158 (2020)
# (but see #5 below)
#

from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

# 1. The only two predefined hybrid functionals as of writing

mc = mcpdft.CASSCF (mf, 'tPBE0', 6, 8).run ()
mc.compute_pdft_energy_(otxc='ftPBE0')

# 2. Other predefined hybrid functionals are not supported

try:
    mc.compute_pdft_energy_(otxc='tB3LYP')
except NotImplementedError as e:
    print ("otxc='tB3LYP' results in NotImplementedError:")
    print (str (e))

# 3. Construct a custom hybrid functional by hand

my_otxc = 't.8*B88 + .2*HF, .8*LYP + .2*HF'
mc.compute_pdft_energy_(otxc=my_otxc)

# 4. Construct the same custom functional using helper function

my_otxc = 't' + mcpdft.hyb ('BLYP', .2)
mc.compute_pdft_energy_(otxc=my_otxc)

# 5. "lambda-MC-PDFT" of JCTC 16, 2274, 2020.

my_otxc = 't' + mcpdft.hyb('PBE', .5, hyb_type='lambda')
#       = 't0.5*PBE + 0.5*HF, 0.75*PBE + 0.5*HF'
mc.compute_pdft_energy_(otxc=my_otxc)




