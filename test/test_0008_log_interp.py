from __future__ import print_function, division
import unittest
import numpy as np

class KnowValues(unittest.TestCase):

  def test_log_interp_sca(self):
    """  """
    from pyscf.nao.m_log_mesh import log_mesh
    from pyscf.nao.m_log_interp import log_interp_c
    rr,pp = log_mesh(1024, 0.01, 20.0)
    log_interp = log_interp_c(rr)
    gc = 1.2030
    ff = np.array([np.exp(-gc*r**2) for r in rr])
    for r in np.linspace(0.009, 25.0, 100):
      y = log_interp(ff, r)
      yrefa = np.exp(-gc*r**2)
      self.assertAlmostEqual(y, yrefa)

  def test_log_interp_vec(self):
    """  """
    from pyscf.nao.m_log_mesh import log_mesh
    from pyscf.nao.m_log_interp import log_interp_c
    rr,pp = log_mesh(1024, 0.01, 20.0)
    log_interp = log_interp_c(rr)
    gcs = np.array([1.2030, 3.2030, 0.7, 10.0])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])
    for r in np.linspace(0.009, 25.0, 100):
      yyref, yy = np.exp(-gcs*r**2), log_interp(ff, r)
      for y,yref in zip(yy, yyref): self.assertAlmostEqual(y,yref)

if __name__ == "__main__":
  unittest.main()
