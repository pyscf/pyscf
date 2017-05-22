from __future__ import print_function, division
import numpy as np

def ls_part_centers(sv, ia1, ia2, ac_rcut):
  """
    For a given atom pair, defined with it's atom indices,
    list all the atoms within a radii
  """
  ls = []
  rmid = (sv.atom2coord[ia1]+sv.atom2coord[ia2])/2.0
  rcut = min(sv.ao_log.sp2rcut[sv.atom2sp[ia1]], sv.ao_log.sp2rcut[sv.atom2sp[ia2]])
  for ia, [ra,sp] in enumerate(zip(sv.atom2coord, sv.atom2sp)):
    #print(ia, ra, sp, rmid, ((ra-rmid)**2).sum())
    if (ac_rcut)**2>((ra-rmid)**2).sum(): ls.append(ia)

  return ls


if __name__ == '__main__':
  from pyscf import gto
  from pyscf.nao import system_vars_c

  mol = gto.M(atom='O 0 0 0; H 0 0 -1; H 0 0 1; H 0 0 2; H 0 0 3;  H 0 0 4; H 0 0 5;', basis='ccpvdz', unit='bohr')
  sv = system_vars_c(gto=mol)
  print(ls_part_centers(sv, 0, 1, 3.0))
  
