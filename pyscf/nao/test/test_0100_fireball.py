from __future__ import print_function, division
import unittest, numpy as np
from pyscf.nao import mf


class KnowValues(unittest.TestCase):

  def test_fireball(self):
    """ Test computation of matrix elements of overlap after fireball """
    sv = mf(fireball="fireball.out", gen_pb=False)
    s_ref = sv.hsx.s4_csr.toarray()
    s = sv.overlap_coo().toarray()
    #print(abs(s-s_ref).sum())

if __name__ == "__main__": unittest.main()

