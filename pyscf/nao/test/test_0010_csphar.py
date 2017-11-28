from __future__ import print_function, division
import unittest
import numpy as np

class KnowValues(unittest.TestCase):

  def test_csphar(self):
    """  """
    from pyscf.nao.m_csphar import csphar
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao, talman2world
    
    rvec = np.array([0.1, 0.2, -0.4])
    lmax = 3
    ylm_py = csphar(rvec, lmax)
    ylm_jt = csphar_talman_libnao(rvec, lmax)
    self.assertEqual(len(ylm_py), (lmax+1)**2)
    self.assertEqual(len(ylm_jt), (lmax+1)**2)
    
    self.assertAlmostEqual(ylm_py[1], 0.075393004386513446-0.15078600877302686j)

    rvecs = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.2, -0.4], [5.1, 2.2, -9.4], [0.9, 0.6, -0.2]]
    for rvec in rvecs:
       ylm_py_ref = csphar(rvec, lmax)
       ylm_py = talman2world(csphar_talman_libnao(rvec, lmax))
       for y1,y2 in zip(ylm_py_ref, ylm_py):
         self.assertAlmostEqual(y1,y2)

if __name__ == "__main__": unittest.main()
