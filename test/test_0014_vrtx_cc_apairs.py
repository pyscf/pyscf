from __future__ import print_function, division
import unittest


class KnowValues(unittest.TestCase):

  def test_vrtx_cc_apairs(self):
    """ This is to test a batch generation vertices for bilocal atomic pairs. """
    from pyscf.nao import system_vars_c, prod_basis_c
    import os

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    #pbb = prod_basis_c().init_prod_basis_pp_batch(sv)
    #pba = prod_basis_c().init_prod_basis_pp(sv)
    
    #print(abs(pbb.get_da2cc_coo().tocsr()-pba.get_da2cc_coo().tocsr()).sum())
    #print(abs(pbb.get_dp_vertex_coo().tocsr()-pba.get_dp_vertex_coo().tocsr()).sum())    
    
if __name__ == "__main__": unittest.main()
