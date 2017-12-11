from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF then G0W0 """
    from pyscf.nao import gw as gw_c
    
    fc = \
      """-1.176136766611261208e+00
      -6.697956300701528587e-01
      -5.155141044813843498e-01
      -4.365449593831648989e-01
      2.104555933716297234e-01
      2.985758300865971604e-01
      5.383650050443907764e-01
      5.960430371457999810e-01
      6.298434571347201194e-01
      6.702163037765542786e-01
      7.488641078920184047e-01
      1.030487843126790093e+00
      1.133598028046180817e+00
      1.308432030099574206e+00
      1.322565372654438409e+00
      1.444842692674101148e+00
      1.831867147891945269e+00
      1.902395044668399482e+00
      1.977107707315529428e+00
      2.119749874222639718e+00
      2.150572413299157493e+00
      2.899028579399293815e+00
      3.912776553395947321e+00"""

    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='water', cd=dname, verbosity=0, nocc_conv=4, nvrt_conv=4, rescf=True)
    gw.kernel_g0w0()
    #np.savetxt('eigvals_g0w0_pyscf_rescf_water_0061.txt', gw.mo_energy_g0w0.T)
      
    for e,eref_str in zip(gw.mo_energy_g0w0,fc.splitlines()):
      self.assertAlmostEqual(e,float(eref_str))


if __name__ == "__main__": unittest.main()
