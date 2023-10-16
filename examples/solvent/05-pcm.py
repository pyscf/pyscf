#!/usr/bin/env python
'''
An example of using PCM solvent models in the mean-field calculations.
'''

from pyscf import gto, scf, dft, cc, solvent, mcscf
from pyscf.solvent import pcm

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

cm = pcm.PCM(mol)
cm.eps = 32.613  # methanol dielectric constant
cm.method = 'C-PCM' # or COSMO, IEF-PCM, SS(V)PE, see https://manual.q-chem.com/5.4/topic_pcm-em.html
cm.lebedev_order = 29 # lebedev grids on the cavity surface, lebedev_order=29  <--> # of grids = 302

# Hartree-Fock
mf = scf.RHF(mol).PCM(cm)
mf.kernel()

g = mf.nuc_grad_method()
grad = g.kernel()

# DFT
mf = dft.RKS(mol, xc='b3lyp').PCM(cm)
mf.kernel()

# calculate gradient of PCM models
g = mf.nuc_grad_method()
grad = g.kernel()

# CCSD, solvent for MP2 can be calcualted in the similar way
mf = scf.RHF(mol) # default method is C-PCM
mf.kernel()
mycc = cc.CCSD(mf).PCM()
mycc.kernel()
de = mycc.nuc_grad_method().as_scanner()(mol)

# CASCI
mf = scf.RHF(mol).PCM()
mf.kernel()
mc = solvent.PCM(mcscf.CASCI(mf,2,2))
de = mc.nuc_grad_method().as_scanner()(mol)

