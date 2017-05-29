from __future__ import print_function, division
import numpy as np

#
#
#
class simulation_c():

  def __init__(self, sv, **kvargs):
    """ Let's try with simulation"""
    self.sv = sv
  
  def do_overlap_check_of_pb(self, **kvargs):
    from pyscf.nao import prod_basis_c, vertex_loop_c
    self.pb = prod_basis_c(self.sv, **kvargs)
    self.vl = vertex_loop_c(self.pb, **kvargs)
    self.mom0,self.mom1 = self.pb.prod_log.comp_moments()
    ad2cc = self.pb.get_ad2cc_den()
    pab2v = self.pb.get_vertex_array()
    

#
# Example of starting a simulation which checks S^ab = V^ab_mu C^mu_nu S^nu
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao import system_vars_c, simulation_c
  import matplotlib.pyplot as plt
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0;', basis='ccpvdz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol, tol=1e-8, nr=512, rmin=1e-5)
  sim = simulation_c(sv)
  sim.do_overlap_check_of_pb()
  
