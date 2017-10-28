from __future__ import print_function, division
import os,unittest
from pyscf.nao import nao

sv = nao(label='water', cd=os.path.dirname(os.path.abspath(__file__)))

class KnowValues(unittest.TestCase):

  def test_lil_vs_coo(self):
    """ Init system variables on libnao's site """
    lil = sv.overlap_lil().tocsr()
    coo = sv.overlap_coo().tocsr()
    derr = abs(coo-lil).sum()/coo.nnz
    self.assertLess(derr, 1e-12)

if __name__ == "__main__": unittest.main()
