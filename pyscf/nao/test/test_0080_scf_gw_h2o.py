from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_scf_gw_perform_h2o(self):
    """ reSCF then G0W0 """
    
    fc = \
      """-1.176137582599898090e+00
-6.697973984258517310e-01
-5.155143130039178123e-01
-4.365448724088398791e-01
2.104535161143837596e-01
2.985738190760626187e-01
5.383631831528181699e-01
5.960427511708059622e-01
6.298425248864513160e-01
6.702150570679562547e-01
7.488635881500678160e-01
1.030485556414411974e+00
1.133596236538136015e+00
1.308430815822860804e+00
1.322564760433334374e+00
1.444841711461231304e+00
1.831867938363858750e+00
1.902393397937107045e+00
1.977107479006525059e+00
2.119748779125555149e+00
2.150570967014801216e+00
2.899024682518652973e+00
3.912773887375614823e+00 """

    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='water', cd=dname, verbosity=0, nocc_conv=4, nvrt_conv=4, perform_scf=True, tol_ia=1e-6)
    gw.kernel_gw()
    np.savetxt('eigvals_g0w0_water_0080.txt', gw.mo_energy_gw[0].T)
      
    for e,eref_str in zip(gw.mo_energy_gw[0,0,:],fc.splitlines()):
      self.assertAlmostEqual(e,float(eref_str))


if __name__ == "__main__": unittest.main()
