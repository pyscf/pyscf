#!/usr/bin/env python
'''
An example of using SMD solvent models in the mean-field calculations.
'''

from pyscf import gto, scf, dft, cc, solvent, mcscf
from pyscf.solvent import smd
from pyscf import hessian

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

# Hartree-Fock with PCM models
mf = scf.RHF(mol).SMD()
mf.with_solvent.solvent = 'water'
mf.kernel()
g = mf.nuc_grad_method()
grad = g.kernel()

# Customize solvent descriptor
mf = scf.RHF(mol).SMD()
# [n, n25, alpha, beta, gamma, epsilon, phi, psi]
mf.with_solvent.solvent_descriptors = [1.3843, 1.3766, 0.0, 0.45, 35.06, 13.45, 0.0, 0.0]
mf.kernel()
g = mf.nuc_grad_method()
grad = g.kernel()
h = mf.Hessian()
hess = h.kernel()

# DFT with PCM models
mf = dft.RKS(mol, xc='b3lyp').SMD()
mf.with_solvent.solvent = 'water'
mf.kernel()
g = mf.nuc_grad_method()      # calculate gradient of DFT with SMD models
grad = g.kernel()
h = mf.Hessian()              # calculate Hessian of DFT with SMD models
hess = h.kernel()

# DFT with density fitting and SMD models
mf = dft.UKS(mol, xc='b3lyp').density_fit().SMD()
mf.with_solvent.solvent = 'water'
g = mf.nuc_grad_method()
grad = mf.kernel()
h = mf.Hessian()
hess = h.kernel()


