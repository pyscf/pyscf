from __future__ import print_function, division

def vnucele_coo_subtract(sv, **kvargs):
  """
  Computes the matrix elements defined by 
    Vne = H_KS - T - V_H - V_xc
  which serve as nuclear-electron attraction matrix elements for pseudo-potential DFT calculations
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  tkin = 0.5*(sv.laplace_coo().tocsr())
  vhar = sv.vhartree_coo(**kvargs).tocsr()
  vxc  = sv.vxc_lil(**kvargs).tocsr()
  vne =  sv.get_hamiltonian(**kvargs)[0].tocsr()-tkin-vhar-vxc
  return vne.tocoo()

