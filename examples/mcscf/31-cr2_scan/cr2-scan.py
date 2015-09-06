#!/usr/bin/env python
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

'''
Scan Cr2 molecule singlet state dissociation curve
'''

ehf = []
emc = []

def run(b, dm, mo, ci=None):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'cr2-%2.1f.out' % b
    mol.atom = [
        ['Cr',(  0.000000,  0.000000, -b/2)],
        ['Cr',(  0.000000,  0.000000,  b/2)],
    ]
    mol.basis = 'cc-pVTZ'
    mol.symmetry = 1
    mol.build()
    m = scf.RHF(mol)
    m.level_shift_factor = .4
    m.max_cycle = 100
    m.conv_tol = 1e-9
    ehf.append(m.scf(dm))

    mc = mcscf.CASSCF(m, 12, 12)
    mc.fcisolver.conv_tol = 1e-9
    if mo is None:
        # the initial guess for b = 1.5
        caslst = [19,20,21,22,23,24,25,28,29,31,32,33]
        mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
    else:
        mo = mcscf.project_init_guess(mc, mo)
    emc.append(mc.kernel(mo)[0])
    mc.analyze()
    return m.make_rdm1(), mc.mo_coeff, mc.ci

dm = mo = ci = None
for b in numpy.arange(1.5, 3.01, .1):
    dm, mo, ci = run(b, dm, mo, ci)

for b in reversed(numpy.arange(1.5, 3.01, .1)):
    dm, mo, ci = run(b, dm, mo, ci)

x = numpy.arange(1.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf2.reverse()
emc2.reverse()
with open('cr2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      HF 3.0->1.5     CAS(12,12)\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,1.5->3.0')
plt.plot(x, ehf2, label='HF,3.0->1.5')
plt.plot(x, emc1, label='CAS(12,12),1.5->3.0')
plt.plot(x, emc2, label='CAS(12,12),3.0->1.5')
plt.legend()
plt.show()
