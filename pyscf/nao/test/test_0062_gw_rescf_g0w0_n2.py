from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ Hartree-Fock than G0W0 N2 example is marked with level-ordering change """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='n2', cd=dname, verbosity=0, jcutoff=9, nff_ia=64, tol_ia=1e-6, rescf=True) 
    gw.kernel_gw()
    gw.report()
    #np.savetxt('eigvals_g0w0_pyscf_rescf_n2_0062.txt', gw.mo_energy_gw.T)

    fc = """-1.294910390463269723e+00
-6.914426700260764003e-01
-5.800631098408213226e-01
-5.800630912944181317e-01
-5.488682018442180288e-01
1.831305221095872460e-01
1.831305925807518165e-01
7.003698201553041347e-01
7.609521815330196892e-01
7.953706485575250396e-01
7.953707042048162590e-01
9.385062725430328712e-01
9.553819617686154508e-01
9.557372926158315130e-01
9.769254942961748123e-01
9.769258344753622980e-01
1.033749364514201741e+00
1.323622829437763881e+00
1.323622831800760569e+00
1.631897154460114852e+00
1.632680402343911652e+00
1.657952682599951544e+00
2.729770515636606998e+00
3.002491147471110899e+00
3.002491923857071310e+00
3.191379493098387865e+00
"""
    for e,eref_str in zip(gw.mo_energy_gw[0,0,:],fc.splitlines()):
      self.assertAlmostEqual(e,float(eref_str))

if __name__ == "__main__": unittest.main()
