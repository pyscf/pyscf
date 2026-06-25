#!/usr/bin/env python
import sys
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import mrpt

verify_windows = '--pyscf-verify-windows' in sys.argv

ehf = []
emc = []
ept = []

def run(b, dm, mo, ci=None):
    if verify_windows:
        mol = gto.M(
            verbose = 5,
            output = 'cr2-%2.1f.out' % b,
            atom = 'Cr 0 0 0; Cr 0 0 %f' % b,
            basis = 'sto-3g',
            symmetry = 0,
        )
        m = scf.RHF(mol)
        m.conv_tol = 1e-9
        ehf.append(m.scf(dm))
        mc = mcscf.CASSCF(m, 6, 6)
        emc.append(mc.kernel()[0])
        ept.append(float('nan'))
        return m.make_rdm1(), mc.mo_coeff, mc.ci

    mol = gto.M(
        verbose = 5,
        output = 'cr2-%2.1f.out' % b,
        atom = 'Cr 0 0 0; Cr 0 0 %f' % b,
        basis = 'sto-3g' if verify_windows else 'cc-pVTZ',
        symmetry = 1,
    )
    m = scf.RHF(mol)
    m.conv_tol = 1e-9
    ehf.append(m.scf(dm))

    mc = mcscf.CASSCF(m, 12, 12)
    if mo is None:
        # the initial guess for b = 1.5
        caslst = [19,20,21,22,23,24,25,28,29,31,32,33]
        mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
    else:
        mo = mcscf.project_init_guess(mc, mo)
    emc.append(mc.kernel(mo)[0])
    mc.analyze()
    if verify_windows:
        # Skip the expensive NEVPT2 step in the installed-wheel verification sweep.
        ept.append(float('nan'))
    else:
        ept.append(mrpt.NEVPT(mc).kernel())
    return m.make_rdm1(), mc.mo_coeff, mc.ci

dm = mo = ci = None
rng = numpy.array([1.5]) if verify_windows else numpy.arange(1.5, 3.01, .1)
for b in rng:
    dm, mo, ci = run(b, dm, mo, ci)

x = rng
with open('cr2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      NEVPT2\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf[i], emc[i], ept[i]))

