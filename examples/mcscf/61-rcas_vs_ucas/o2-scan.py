#!/usr/bin/env python
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf


def run(b, mo0=None, dm0=None):
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'o2rhf-%3.2f.out' % b,
        atom = [
            ['O', (0, 0,  b/2)],
            ['O', (0, 0, -b/2)],],
        basis = 'cc-pvdz',
        spin = 2,
        symmetry = 1,
    )

    mf = scf.RHF(mol)
    mf.scf(dm0)

    mc = mcscf.CASSCF(mf, 12, 8)
    if mo0 is not None:
        #from pyscf import lo
        #mo0 = lo.orth.vec_lowdin(mo0, mf.get_ovlp())
        mo0 = mcscf.project_init_guess(mc, mo0)
    else:
        mo0 = mcscf.sort_mo(mc, mf.mo_coeff, [5,6,7,8,9,11,12,13,14,15,16,17])
        mc.max_orb_stepsize = .02
    mc.kernel(mo0)
    mc.analyze()
    return mf, mc


def urun(b, mo0=None, dm0=None):
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'o2uhf-%3.2f.out' % b,
        atom = [
            ['O', (0, 0,  b/2)],
            ['O', (0, 0, -b/2)],],
        basis = 'cc-pvdz',
        spin = 2,
    )

    mf = scf.UHF(mol)
    mf.scf(dm0)

    mc = mcscf.CASSCF(mf, 12, 8)
    if mo0 is not None:
        #from pyscf import lo
        #mo0 =(lo.orth.vec_lowdin(mo0[0], mf.get_ovlp()),
        #      lo.orth.vec_lowdin(mo0[1], mf.get_ovlp()))
        mo0 = mcscf.project_init_guess(mc, mo0)
    mc.kernel(mo0)
    mc.analyze()
    return mf, mc

x = numpy.hstack((numpy.arange(0.9, 2.01, 0.1),
                  numpy.arange(2.1, 4.01, 0.1)))

dm0 = mo0 = None
eumc = []
euhf = []
s = []
for b in reversed(x):
    mf, mc = urun(b, mo0, dm0)
    mo0 = mc.mo_coeff
    dm0 = mf.make_rdm1()
    s.append(mc.spin_square()[1])
    euhf.append(mf.hf_energy)
    eumc.append(mc.e_tot)

euhf.reverse()
eumc.reverse()
s.reverse()
#print s

dm0 = mo0 = None
ermc = []
erhf = []
for b in x:
    mf, mc = run(b, mo0, dm0)
    mo0 = mc.mo_coeff
    dm0 = mf.make_rdm1()
    erhf.append(mf.hf_energy)
    ermc.append(mc.e_tot)

with open('o2-scan.txt', 'w') as fout:
    fout.write('  ROHF 0.9->4.0   RCAS(12,8)    UHF 4.0->0.9  UCAS(12,8) \n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, erhf[i], ermc[i], euhf[i], eumc[i]))

import matplotlib.pyplot as plt
plt.plot(x, erhf, label='ROHF,0.9->4.0')
plt.plot(x, euhf, label='UHF, 4.0->0.9')
plt.plot(x, ermc, label='RCAS(6,6),0.9->4.0')
plt.plot(x, eumc, label='UCAS(6,6),4.0->0.9')
plt.legend()
plt.show()
