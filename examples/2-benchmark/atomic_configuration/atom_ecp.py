#!/usr/bin/env python

'''
Benchmark atomic configuration regarding to issue #518

Atomic calculation with ECP and ECP basis using symmetry averaged occupancy
'''

from pyscf import gto
from pyscf.data import elements
from pyscf.scf import atom_hf

# RHF
atoms = {}
for z in range(21, 94):
    try:
        a = gto.M(atom=[[z, (0, 0, 0)]], basis='lanl2dz', ecp='lanl2dz', verbose=0, spin=None)
        atoms[z] = a
    except:
        pass

def count_scf_cycle(envs):
    envs['mf']._cycle = envs['cycle']

counts = {}
for z, atm in atoms.items():
    mf = atom_hf.AtomSphericAverageRHF(atm)
    mf.atomic_configuration = elements.NRSRHF_CONFIGURATION
    mf.callback = count_scf_cycle
    mf.run()

    mf1 = atom_hf.AtomSphericAverageRHF(atm)
    mf1.atomic_configuration = elements.CONFIGURATION
    mf1.callback = count_scf_cycle
    mf1.run()
    counts[z] = (mf._cycle, mf.e_tot, mf1._cycle, mf1.e_tot)

print('E(NRSRHF) is lower in %d samples' % sum([(v[1] - v[3] < 1e-7) for v in counts.values()]))
print('E(NRSRHF) is higher in %d samples' % sum([(v[1] - v[3] > 1e-7) for v in counts.values()]))

