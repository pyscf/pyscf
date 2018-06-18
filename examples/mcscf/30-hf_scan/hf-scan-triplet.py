#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import mcscf

'''
Scan HF molecule triplet state dissociation curve
'''

ehf = []
emc = []

def run(b, dm, mo):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_hf-%2.1f' % b
    mol.atom = [
        ["F", (0., 0., 0.)],
        ["H", (0., 0., b)],]
    mol.spin = 2
    mol.basis = {'F': 'cc-pvdz',
                 'H': 'cc-pvdz',}
    mol.build()
    m = scf.RHF(mol)
    ehf.append(m.scf(dm))

    mc = mcscf.CASSCF(m, 6, 6)
    if mo is None:
        mo = mcscf.sort_mo(mc, m.mo_coeff, [3,4,5,6,9,10])
    else:
        mo = mcscf.project_init_guess(mc, mo)
    e1 = mc.mc1step(mo)[0]
    emc.append(e1)
    return m.make_rdm1(), mc.mo_coeff

dm = mo = None
for b in numpy.arange(0.7, 4.01, 0.1):
    dm, mo = run(b, dm, mo)

for b in reversed(numpy.arange(0.7, 4.01, 0.1)):
    dm, mo = run(b, dm, mo)

x = numpy.arange(0.7, 4.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf2.reverse()
emc2.reverse()
with open('hf-scan-triplet.txt', 'w') as fout:
    fout.write('    HF 4.0->0.7    CAS(6,6)      HF 0.7->4.0   CAS(6,6)  \n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, emc1, label='CAS(6,6),4.0->0.7')
plt.plot(x, emc2, label='CAS(6,6),0.7->4.0')
plt.legend()
plt.show()
