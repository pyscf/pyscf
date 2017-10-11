from __future__ import print_function, division

def vnucele_coo(sv, algo=None, **kvargs):
  """
  Computes the nucleus-electron attraction matrix elements
  Args:
    sv : (System Variables), this must have arrays of coordinates and species, etc
  Returns:
    matrix elements

      These are tricky to define. In case of all-electron calculations it is well known, but 
      in case of pseudo-potential calculations we need some a specification of pseudo-potential
      and a specification of (Kleinman-Bulander) projectors to compute explicitly. Practically,
      we will subtract the computed matrix elements from the total Hamiltonian to find out the 
      nuclear-electron interaction in case of SIESTA import. This means that Vne is defined by 
        Vne = H_KS - T - V_H - V_xc

  """
  if algo is None: 
    if hasattr(sv, 'xml_dict'): 
      vne_coo = sv.vnucele_coo_subtract(**kvargs) # try to subtract if data is coming from SIESTA
    else:
      vne_coo = sv.vnucele_coo_coulomb(**kvargs)  # try to compute the Coulomb attraction matrix elements
  else:
    vne_coo = sv.algo(**kvargs)

  return vne_coo

