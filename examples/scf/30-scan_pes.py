#!/usr/bin/env python

'''
Scan HF/DFT PES.
'''

import numpy
from pyscf import gto
from pyscf import scf, dft

#
# A scanner can take the initial guess from previous calculation
# automatically.
#
mol = gto.Mole()
mf_scanner = scf.RHF(mol).as_scanner()
ehf1 = []
for b in numpy.arange(0.7, 4.01, 0.1):
    mol = gto.M(verbose = 5,
                output = 'out_hf-%2.1f' % b,
                atom = [["F", (0., 0., 0.)],
                        ["H", (0., 0., b)],],
                basis = 'cc-pvdz')
    ehf1.append(mf_scanner(mol))

#
# Create a new scanner, the results of last calculation will not be used as
# initial guess.
#
mf_scanner = dft.RKS(mol).set(xc='b3lyp').as_scanner()
ehf2 = []
for b in reversed(numpy.arange(0.7, 4.01, 0.1)):
    # Scanner supports to input the structure of a molecule than the Mole object
    ehf2.append(mf_scanner([["F", (0., 0., 0.)],
                            ["H", (0., 0., b)],]))

x = numpy.arange(0.7, 4.01, .1)
ehf2.reverse()
with open('hf-scan.txt', 'w') as fout:
    fout.write('       HF 0.7->4.0    B3LYP 4.0->0.7\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %14.8f  %14.8f\n'
                   % (xi, ehf1[i], ehf2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,0.7->4.0')
plt.plot(x, ehf2, label='HF,4.0->0.7')
plt.legend()
plt.show()
