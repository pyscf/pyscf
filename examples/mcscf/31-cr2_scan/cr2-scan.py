#!/usr/bin/env python

'''
Scan Cr2 molecule singlet state dissociation curve.

Simliar tthe example mcscf/30-hf_scan, we need to control the CASSCF initial
guess using functions project_init_guess and sort_mo.  In this example,
sort_mo function is replaced by the symmetry-adapted version
``sort_mo_by_irrep``.
'''

import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

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
    mf = scf.RHF(mol)
    mf.level_shift = .4
    mf.max_cycle = 100
    mf.conv_tol = 1e-9
    ehf.append(mf.scf(dm))

    mc = mcscf.CASSCF(mf, 12, 12)
    mc.fcisolver.conv_tol = 1e-9
    # FCI solver with multi-threads is not stable enough for this sytem
    mc.fcisolver.threads = 1
    if mo is None:
        # the initial guess for b = 1.5
        ncore = {'A1g':5, 'A1u':5}  # Optional. Program will guess if not given
        ncas = {'A1g':2, 'A1u':2,
                'E1ux':1, 'E1uy':1, 'E1gx':1, 'E1gy':1,
                'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}
        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    else:
        mo = mcscf.project_init_guess(mc, mo)
    emc.append(mc.kernel(mo, ci)[0])
    mc.analyze()
    return mf.make_rdm1(), mc.mo_coeff, mc.ci

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
