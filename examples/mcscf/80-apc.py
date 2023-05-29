#!/usr/bin/env python

# Author: Daniel S. King

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import apc

mol = gto.Mole()
mol.atom = [('O', [0.0, 0.0, -0.13209669380597672]),
            ('H', [0.0, 1.4315287853817316, 0.9797000689025815]),
            ('H', [0.0, -1.4315287853817316, 0.9797000689025815])]
mol.basis = "6-31g"
mol.unit = "bohr"
mol.build()
mf = scf.RHF(mol)
mf.kernel()

#Choose a max(10,10) active space with APC entropies:
myapc = apc.APC(mf,max_size=(10,10))
ncas,nelecas,casorbs = myapc.kernel()

#Choose a 12-orbital active space with APC entropies:
myapc = apc.APC(mf,max_size=12)
ncas,nelecas,casorbs = myapc.kernel()
    
#Use custom entropies with Chooser class
entropies = np.random.choice(np.arange(len(mf.mo_occ)),len(mf.mo_occ),replace=False) #randomly ranked orbitals
chooser = apc.Chooser(mf.mo_coeff,mf.mo_occ,entropies,max_size=(8,8))
ncas,nelecas,casorbs = myapc.kernel()
    
#Use custom orbitals with APC entropies by modifying mf.mo_coeff
#APC entropies will still be calculated with info from mf.mo_occ, mf.get_fock, and mf.get_k:
mf2 = scf.RKS(mol) #example: dft MOs
mf2.kernel()
mf.mo_coeff = mf2.mo_coeff
myapc = apc.APC(mf,max_size=6)
ncas,nelecas,casorbs = myapc.kernel()

#Use any of these selected active spaces in CASCI/CASSCF:
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.mo_coeff = casorbs
mc.kernel()


