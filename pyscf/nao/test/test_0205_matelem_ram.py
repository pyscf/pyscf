from __future__ import print_function, division
import os, unittest, numpy as np

from timeit import default_timer as timer

from pyscf.nao import mf as mf_c
from pyscf.nao.m_ao_eval import ao_eval
    
class KnowValues(unittest.TestCase):


  def test_matelem_speed(self):
    """ Test the computation of atomic orbitals in coordinate space """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(verbosity=0, label='water', cd=dname, gen_pb=False, force_gamma=True, Ecut=50)
    g = mf.mesh3d.get_3dgrid()
    t0 = timer()
    vna = mf.vna(g.coords)
    t1 = timer()
    ab2v1 = mf.matelem_int3d_coo(g, vna)
    t2 = timer()
    ab2v2 = mf.matelem_int3d_coo_ref(g, vna)
    t3 = timer()
    #print(__name__, 't1 t2: ', t1-t0, t2-t1, t3-t2)
    #print(abs(ab2v1.toarray()-ab2v2.toarray()).sum()/ab2v2.size, (abs(ab2v1.toarray()-ab2v2.toarray()).max()))
        
    self.assertTrue(np.allclose(ab2v1.toarray(), ab2v2.toarray()))


if __name__ == "__main__": unittest.main()
