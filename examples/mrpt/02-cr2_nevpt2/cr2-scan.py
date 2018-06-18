#!/usr/bin/env python
import numpy
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import lo
from pyscf import mrpt

ehf = []
emc = []
ept = []

def run(b, dm, mo, ci=None):
    mol = gto.M(
        verbose = 5,
        output = 'cr2-%2.1f.out' % b,
        atom = 'Cr 0 0 0; Cr 0 0 %f' % b,
        basis = 'cc-pVTZ',
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
    ept.append(mrpt.NEVPT(mc).kernel())
    return m.make_rdm1(), mc.mo_coeff, mc.ci

dm = mo = ci = None
for b in numpy.arange(1.5, 3.01, .1):
    dm, mo, ci = run(b, dm, mo, ci)

x = numpy.arange(1.5, 3.01, .1)
with open('cr2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      NEVPT2\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf[i], emc[i], ept[i]))

