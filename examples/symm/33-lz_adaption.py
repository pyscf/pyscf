#!/usr/bin/python

'''
This example shows how to construct pure lz adapated basis.
'''

import numpy
from pyscf import gto, scf

def lz_adapt_(mol):
    print('mol.irrep_id', mol.irrep_id)
    A_irrep_ids = set([0,1,4,5])
    E_irrep_ids = set(mol.irrep_id).difference(A_irrep_ids)
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir%10) in (0,2,5,7)]

# See L146 of pyscf/symm/basis.py
    for k, ir in enumerate(Ex_irrep_ids):
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = mol.irrep_id.index(ir)
            Ey = mol.irrep_id.index(ir+1)
        else:
            Ex = mol.irrep_id.index(ir)
            Ey = mol.irrep_id.index(ir-1)

        if ir % 10 in (0, 5):
            lz = (ir // 10) * 2
        else:
            lz = (ir // 10) * 2 + 1

        print('Transforming Ex %d Ey %d    Lz = %d' %
              (mol.irrep_id[Ex], mol.irrep_id[Ey], lz))
        lz_minus = numpy.sqrt(.5) * (mol.symm_orb[Ex] - mol.symm_orb[Ey] * 1j)
        if lz % 2 == 0:
            lz_plus = numpy.sqrt(.5) * (mol.symm_orb[Ex] + mol.symm_orb[Ey] * 1j)
        else:
            lz_plus = -numpy.sqrt(.5) * (mol.symm_orb[Ex] + mol.symm_orb[Ey] * 1j)
        mol.symm_orb[Ey] = lz_minus  # x-iy
        mol.symm_orb[Ex] = lz_plus   # x+iy
    return mol

mol = gto.M(atom='Ne', basis='ccpvtz', symmetry='d2h')
mol = lz_adapt_(mol)
mf = scf.RHF(mol)
mf.run()
