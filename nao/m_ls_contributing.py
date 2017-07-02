from __future__ import print_function, division
import numpy as np
from numpy import sqrt

def ls_contributing(pb, sp12, ra12):
  """
  List of contributing centers 
  prod_basis_c : instance of the prod_basis_c containing parameters .ac_rcut_ratio and .ac_npc_max and .sv providing instance of system_vars_c which provides the coordinates, unit cell vectors, species etc. and .prod_log prividing information on the cutoffs for each specie.
  sp12 : a couple of species
  ra12 : a couple of coordinates, correspondingly
  """
  sv = pb.sv
  pl = pb.prod_log

  ls = []
  ra3 = 0.5*(ra12[0,:] + ra12[1,:])
  for sp, rvec in zip(sv.atom2sp,sv.atom2coord):
    rc = pl.sp2rcut[sp]
    #print(rc-sqrt(sum((ra12[0,:]-rvec)**2)), rc-sqrt(sum((ra12[1,:]-rvec)**2)) )

