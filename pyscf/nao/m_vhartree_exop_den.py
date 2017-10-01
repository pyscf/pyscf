from __future__ import print_function, division

def vhartree_exop_den(sv, dm=None, **kvargs):
  """
  Computes the matrix elements of Hartree potential and Fock exchange potential
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """  
  dm = sv.comp_dm() if dm is None else dm
  vh = sv.vhartree_coo(dm=dm).todense()
  jmat = sv.jmat_den(dm=dm)
  return vh,jmat

