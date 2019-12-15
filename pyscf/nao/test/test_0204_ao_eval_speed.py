from __future__ import print_function, division
import os, unittest, numpy as np

from timeit import default_timer as timer

from pyscf.nao import mf as mf_c
from pyscf.nao.m_ao_eval import ao_eval
    
class KnowValues(unittest.TestCase):


  def test_ao_eval_speed(self):
    """ Test the computation of atomic orbitals in coordinate space """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(verbosity=0, label='water', cd=dname, gen_pb=False, force_gamma=True, Ecut=20)
    g = mf.mesh3d.get_3dgrid()
    t0 = timer()
    oc2v1 = mf.comp_aos_den(g.coords)
    t1 = timer()
    oc2v2 = mf.comp_aos_py(g.coords)
    t2 = timer()
    
    print(__name__, 't1 t2: ', t1-t0, t2-t1)
    
    print(abs(oc2v1-oc2v2).sum()/oc2v2.size, (abs(oc2v1-oc2v2).max()))
        
    self.assertTrue(np.allclose(oc2v1, oc2v2, atol=3.5e-5))


if __name__ == "__main__": unittest.main()
