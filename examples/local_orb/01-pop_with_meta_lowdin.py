#!/usr/bin/env python

'''
Mulliken population analysis with meta-Lowdin orbitals
'''

import numpy
from pyscf import gto, scf, lo

x = .63
mol = gto.M(atom=[['C', (0, 0, 0)],
                  ['H', (x ,  x,  x)],
                  ['H', (-x, -x,  x)],
                  ['H', (-x,  x, -x)],
                  ['H', ( x, -x, -x)]],
            basis='ccpvtz')
mf = scf.RHF(mol).run()

c = lo.orth_ao(mol, 'meta_lowdin')
mo = numpy.linalg.solve(c, mf.mo_coeff)
dm = mf.make_rdm1(mo, mf.mo_occ)
mf.mulliken_pop(mol, dm, numpy.eye(mol.nao_nr()))
