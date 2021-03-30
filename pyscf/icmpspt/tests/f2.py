#!/usr/bin/env python
from pyscf import gto
mol = gto.Mole()
mol.verbose = 4
mol.output = None
mol.atom = [
        ['F',(0.0000000000,0.0000000000,-0.70595)], ['F',(0.0000000000,0.0000000000,0.70595)], 
]
mol.basis = 'cc-pvdz'
mol.spin = 0
mol.build()

from pyscf import scf
m = scf.RHF(mol)
m.verbose = 4
ehf = m.scf()
print("EHF",ehf)

from pyscf import cc
mcc = cc.CCSD(m,frozen=0)
mcc.kernel()
etriple=mcc.ccsd_t()
print("ECCSD",ehf+mcc.e_corr)
print("ECCSDT",ehf+mcc.e_corr+etriple)

from pyscf.shciscf import shci
mc = shci.SHCISCF(m, 8,14, frozen=2)
mc.fcisolver.mpiprefix="mpirun -np 28"
mc.fcisolver.num_thrds=12
#mc.fcisolver.sweep_iter    = [   5]
#mc.fcisolver.sweep_epsilon = [1e-5]
#mc.max_cycle_macro=0
mc.verbose = 4
ci_e = mc.kernel()[0]
print("ECAS",ci_e)

from pyscf.icmpspt import icmpspt
energy=icmpspt.mrlcc(mc,nfro=0)
print("ETOT",energy)
