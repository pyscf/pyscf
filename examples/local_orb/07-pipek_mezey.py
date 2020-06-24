#!/usr/bin/env python

'''
Various Pipek-Mezey localization schemes

See more discussions in paper  JCTC, 10, 642 (2014); DOI:10.1021/ct401016x
'''

import pyscf

x = .63
mol = pyscf.M(atom=[['C', (0, 0, 0)],
                    ['H', (x ,  x,  x)],
                    ['H', (-x, -x,  x)],
                    ['H', (-x,  x, -x)],
                    ['H', ( x, -x, -x)]],
              basis='ccpvtz')

mf = mol.RHF().run()

orbitals = mf.mo_coeff[:,mf.mo_occ>0]
pm = pyscf.lo.PM(mol, orbitals, mf)
def print_coeff(local_orb):
    import sys
    idx = mol.search_ao_label(['C 1s', 'C 2s', 'C 2p', 'H 1s'])
    ao_labels = mol.ao_labels()
    labels = [ao_labels[i] for i in idx]
    pyscf.tools.dump_mat.dump_rec(sys.stdout, local_orb[idx], labels)

print('---')
pm.pop_method = 'mulliken'
print_coeff(pm.kernel())

print('---')
pm.pop_method = 'meta-lowdin'
print_coeff(pm.kernel())

print('---')
pm.pop_method = 'iao'
print_coeff(pm.kernel())

print('---')
pm.pop_method = 'becke'
print_coeff(pm.kernel())
