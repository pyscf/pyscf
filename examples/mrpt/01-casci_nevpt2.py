#!/usr/bin/env python

"""Strongly contracted NEVPT2 from CASCI references for O2 and an H14 chain.

The active orbitals are taken from HF, without CASSCF orbital optimization.
The printed reference corrections are retained from the original nevpt2.py
module demonstration; they are not assertions. The H14 calculation is more
expensive because of its larger active space.
"""

from pyscf import gto
from pyscf import scf
from pyscf import mcscf, mrpt

# Triplet O2/6-31G: eight active electrons in six orbitals.
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', ( 0., 0.    , 0.    )],
    ['O', ( 0., 0.    , 1.207 )],
]
mol.basis = '6-31g'
mol.spin = 2
mol.build()

m = scf.RHF(mol)
m.conv_tol = 1e-20
ehf = m.scf()
mc = mcscf.CASCI(m, 6, 8)
mc.fcisolver.conv_tol = 1e-14
ci_e = mc.kernel()[0]
mc.verbose = 4
print('O2 CASCI total energy:', ci_e)
e_corr = mrpt.NEVPT(mc).kernel()
print('O2 NEVPT2 correction and original reference:', e_corr, -0.169785157128082)
print('O2 CASCI + NEVPT2 total energy:', mc.e_tot + e_corr)


# Alternating H14 chain/STO-3G: ten active electrons in eight orbitals.
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H', ( 0., 0.    , 0.    )],
    ['H', ( 0., 0.    , 0.8   )],
    ['H', ( 0., 0.    , 2.    )],
    ['H', ( 0., 0.    , 2.8   )],
    ['H', ( 0., 0.    , 4.    )],
    ['H', ( 0., 0.    , 4.8   )],
    ['H', ( 0., 0.    , 6.    )],
    ['H', ( 0., 0.    , 6.8   )],
    ['H', ( 0., 0.    , 8.    )],
    ['H', ( 0., 0.    , 8.8   )],
    ['H', ( 0., 0.    , 10.    )],
    ['H', ( 0., 0.    , 10.8   )],
    ['H', ( 0., 0.    , 12     )],
    ['H', ( 0., 0.    , 12.8   )],
]
mol.basis = {'H': 'sto-3g'}
mol.build()

m = scf.RHF(mol)
m.conv_tol = 1e-20
ehf = m.scf()
mc = mcscf.CASCI(m,8,10)
mc.fcisolver.conv_tol = 1e-14
mc.kernel()
mc.verbose = 4
e_corr = mrpt.NEVPT(mc).kernel()
print('H14 NEVPT2 correction and original reference:', e_corr, -0.094164359938171)
print('H14 CASCI + NEVPT2 total energy:', mc.e_tot + e_corr)
