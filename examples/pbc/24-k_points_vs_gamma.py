#!/usr/bin/env python

'''
Showing use of the parallelized CCSD with K-point sampling.
'''

import numpy as np
from pyscf.pbc import cc as pbccc
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto
from pyscf.pbc.tools.pbc import super_cell

nmp = [1, 1, 2]
cell = gto.M(
    unit='B',
    a=[[0., 3.37013733, 3.37013733],
       [3.37013733, 0., 3.37013733],
       [3.37013733, 3.37013733, 0.]],
    mesh=[24,]*3,
    atom='''C 0 0 0
              C 1.68506866 1.68506866 1.68506866''',
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4
)

# We build a supercell composed of 'nmp' replicated units and run
# our usual molecular Hartree-Fock program, but using integrals
# between periodic gaussians.
#cell = build_cell(ase_atom, ke=50., basis=basis)
supcell = super_cell(cell, nmp)
mf = pbchf.RHF(supcell)
mf.kernel()
supcell_energy = mf.energy_tot() / np.prod(nmp)

# A wrapper calling molecular CC method for gamma point calculation.
mycc = pbccc.RCCSD(mf)
gccsd_energy = mycc.ccsd()[0] / np.prod(nmp)
eip, wip = mycc.ipccsd(nroots=2)
eea, wea = mycc.eaccsd(nroots=2)

# We now begin our k-point calculations for the same system, making
# sure we shift the k-points to be gamma-centered.
kpts = cell.make_kpts(nmp)
kpts -= kpts[0]
kmf = pbchf.KRHF(cell, kpts)
kpoint_energy = kmf.kernel()

mykcc = pbccc.KRCCSD(kmf)
kccsd_energy = mykcc.ccsd()[0]
ekcc = mykcc.ecc
# We look at a gamma-point transition for IP/EA
ekip, wkip = mykcc.ipccsd(nroots=2, kptlist=[0])
ekea, wkea = mykcc.eaccsd(nroots=2, kptlist=[0])

print('Difference between gamma/k-point mean-field calculation = %.15g' % (
    abs(supcell_energy-kpoint_energy)))
print('Difference between gamma/k-point ccsd calculation = %.15g' % (
    abs(gccsd_energy - kccsd_energy)))
print('Difference between gamma/k-point ip-eomccsd calculation = %.15g' % (
    np.linalg.norm(np.array(eip) - np.array(ekip))))
print('Difference between gamma/k-point ea-eomccsd calculation = %.15g' % (
    np.linalg.norm(np.array(eea) - np.array(ekea))))
