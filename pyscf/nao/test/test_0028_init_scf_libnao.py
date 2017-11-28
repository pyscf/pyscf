from __future__ import print_function, division
import os,unittest
from pyscf.nao import mf

sv = mf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))

class KnowValues(unittest.TestCase):

  def test_init_sv_libnao(self):
    """ Init system variables on libnao's site """
    sv.init_libnao()
    

if __name__ == "__main__": unittest.main()
