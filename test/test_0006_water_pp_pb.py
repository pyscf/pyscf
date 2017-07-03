from __future__ import print_function, division
import unittest

class KnowValues(unittest.TestCase):

  def test_water_pp_pb(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao import system_vars_c, prod_basis_c
    from numpy import einsum, array
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(chdir=dname)
    self.assertTrue(abs(sv.ucell).sum()>0)
    pb = prod_basis_c().init_pb_pp_libnao_apair(sv)
    self.assertEqual(sv.norbs, 23)
    pb.init_prod_basis_pp()
    self.assertEqual(len(pb.bp2info), 3)
    vden = pb.get_vertex_array()
    ccden = pb.get_da2cc_den()
    moms = pb.comp_moments()

    oref = sv.overlap_coo().toarray()
    over = einsum('lab,l->ab', vden, moms[0])

    dcoo = sv.dipole_coo()
    dref = array([dc.toarray() for dc in dcoo])
    dipo = einsum('lab,lx->xab', vden, moms[1])
    
    #print(abs(oref-over).sum()/(oref.size))
    #print(abs(oref-over).max())

    #print(abs(dref-dipo).sum()/(dref.size))
    #print(abs(dref-dipo).max())
    
#    print(vden.shape)
#    print(ccden.shape)
#    print(moms[0].shape)
#    print(moms[1].shape)
    

if __name__ == "__main__":
  unittest.main()
