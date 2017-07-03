from __future__ import print_function, division
import unittest
from pyscf import gto
import os

d = os.path.dirname(os.path.abspath(__file__))
mol = gto.M(
    verbose = 1,
    atom = open(d+"/ag_s7l7_wonatoms.xyz").read()
)

class KnowValues(unittest.TestCase):

  def test_ls_contributing(self):
    """ To test the list of contributing centers """
    from pyscf.nao import system_vars_c, prod_basis_c
    sv = system_vars_c().init_pyscf_gto(mol)
    pb = prod_basis_c()
    pb.sv = sv
    pb.sv.ao_log.sp2rcut[0] = 10.0
    pb.prod_log = sv.ao_log
    pb.prod_log.sp2rcut[0] = 10.0
    pb.ac_rcut = max(sv.ao_log.sp2rcut)
    pb.ac_npc_max = 10
    lsc = pb.ls_contributing(0,1)
    self.assertEqual(len(lsc),10)
    lsref = [ 0,  1, 13,  7,  5, 43, 42, 39, 38, 10]
    for i,ref in enumerate(lsref) : self.assertEqual(lsc[i],ref)
    

    
if __name__ == "__main__":
  unittest.main()
