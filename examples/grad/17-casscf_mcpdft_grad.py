#!/usr/bin/env python

'''
Analytical nuclear gradients of CASSCF-based MC-PDFT
(gradients for CASCI-based MC-PDFT not available)
'''

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz',
    verbose = 5)

mf = scf.RHF(mol).run()
mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4).run()
mc_grad = mc.nuc_grad_method ().run ()
