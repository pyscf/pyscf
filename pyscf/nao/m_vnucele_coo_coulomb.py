from __future__ import print_function, division

def vnucele_coo_coulomb(sv, **kvargs):
  """
  Computes the matrix elements defined by 
    Vne = f(r) sum_a   Z_a/|r-R_a|  g(r)
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements
  """
  from numpy import einsum, dot
  from scipy.sparse import coo_matrix
  g = sv.build_3dgrid_ae(**kvargs)
  ca2o = sv.comp_aos_den(g.coords)
  vnuc = sv.comp_vnuc_coulomb(g.coords)
  vnuc_w = g.weights*vnuc
  cb2vo = einsum('co,c->co', ca2o, vnuc_w)
  vne = dot(ca2o.T,cb2vo)
  return coo_matrix(vne)

