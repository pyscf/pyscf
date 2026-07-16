#!/usr/bin/env python

'''
Analytical nuclear gradients of CASCI with HF and restricted KS orbitals.
'''

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import mcscf


mol = gto.M(
    atom = '''
    H  0.000  0.000  0.000
    H  0.000  0.000  1.000
    H  0.200  1.050  0.000
    ''',
    basis = 'sto-3g',
    spin = 1,
    verbose = 0,
)

mf = scf.UHF(mol).run()
mc = mcscf.UCASCI(mf, 2, (1, 1), ncore=(1, 0)).run()

de = mc.nuc_grad_method().kernel()
print('UHF-orbital UCASCI gradients')
print(de)

mc.fcisolver.nroots = 3
mc.kernel()
de = mc.nuc_grad_method().kernel(state=1)
print('UHF-orbital UCASCI gradients for state 1')
print(de)


mol = gto.M(
    atom = '''
    H  0.000  0.000  0.000
    H  0.000  0.000  1.000
    H  0.000  1.000  0.000
    H  0.200  1.000  1.000
    ''',
    basis = 'sto-3g',
    verbose = 0,
)

mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 2, 2, ncore=1).run()

de = mc.Gradients().kernel()
print('RHF-orbital CASCI gradients')
print(de)

mf = dft.RKS(mol, xc='b3lyp').run()
mc = mcscf.CASCI(mf, 2, 2, ncore=1).run()

de = mc.Gradients().kernel()
print('RKS-orbital CASCI gradients')
print(de)
