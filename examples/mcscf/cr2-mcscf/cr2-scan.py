#!/usr/bin/env python
import numpy
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.tools import dump_mat

ehf = []
emc = []

def run(b, caslst):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'cr2-%2.1f.out' % b
    mol.atom = [
        ['Cr',(  0.000000,  0.000000, -b/2)],
        ['Cr',(  0.000000,  0.000000,  b/2)],
    ]
    mol.charge = 0
    mol.basis = {'Cr': 'cc-pVTZ', }
    mol.build()
    m = scf.RHF(mol)
    m.chkfile = 'cr2.chk'
    m.init_guess = 'chkfile'
    m.level_shift_factor = .5
    m.get_occ = scf.addons.frac_occ(m)
    m.diis_space = 25
    m.max_cycle = 100
    m.conv_tol = 1e-9
    ehf.append(m.scf())

    mc = mcscf.CASSCF(mol, m, 12, 12)
    mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
    emc.append(mc.mc1step(mo)[0]+mol.energy_nuc())

    label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
    dm1a, dm1b = mcscf.addons.make_rdm1s(mc, mc.ci, mc.mo_coeff)
    dump_mat.dump_tri(m.stdout, dm1a, label)

    mcscf.addons.map2hf(mc)
    return ehf, emc

for b in 3.0, 2.9, 2.8:
    caslst = [19,20,21,22,23,24,31,32,33,34,35,40]
    run(b, caslst)

for b in 2.7, 2.6, 2.5, 2.4:
    caslst = [19,20,21,22,23,24,30,31,33,34,35,40]
    run(b, caslst)

for b in 2.3, 2.2, 2.1:
    caslst = [19,20,21,22,23,24,27,28,29,34,35,40]
    run(b, caslst)

for b in 2.0, 1.9, 1.8, 1.7:
    caslst = [19,20,21,22,23,24,27,28,33,34,35,40]
    run(b, caslst)

for b in 1.6,:
    caslst = [19,20,21,22,23,24,28,29,33,34,35,40]
    run(b, caslst)

for b in 1.5,:
    caslst = [19,20,21,22,23,24,25,26,27,32,37,38]
    run(b, caslst)

import os
os.remove('cr2.chk')

for b in 1.5,:
    caslst = [19,20,21,22,23,24,25,28,29,30,32,33]
    run(b, caslst)

for b in 1.6, 1.7, 1.8, 1.9, 2.0:
    caslst = [19,20,21,22,23,24,25,28,29,30,31,32]
    run(b, caslst)

for b in 2.1, 2.2, 2.3, 2.4:
    caslst = [19,20,21,22,23,24,25,26,27,28,31,32]
    run(b, caslst)

for b in 2.5, 2.6, 2.7, 2.8, 2.9, 3.0:
    caslst = [19,20,21,22,23,24,25,26,27,28,29,30]
    run(b, caslst)

x = numpy.arange(1.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf1.reverse()
emc1.reverse()
with open('cr2-scan.txt', 'w') as fout:
    fout.write('     HF-A            CAS(12,12)-A    HF-B            CAS(12,12)-B\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,3.0->1.5')
plt.plot(x, ehf2, label='HF,1.5->3.0')
plt.plot(x, emc1, label='CAS(12,12),3.0->1.5')
plt.plot(x, emc2, label='CAS(12,12),1.5->3.0')
plt.legend()
plt.show()
