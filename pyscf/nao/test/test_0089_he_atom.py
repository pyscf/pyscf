from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''He 0.0 0.0 0.0''', basis = 'aug-cc-pvtz', spin = 0, )
gto_mf = scf.UHF(mol)
e_tot = gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_0089_he_atom(self):
    from io import StringIO
    """ Spin-resolved case GW procedure. """
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=1, niter_max_ev=16, rescf=True, kmat_algo='dp_vertex_loops_sm')
    self.assertEqual(gw.nspin, 2)
    gw.kernel_gw()
    gw.report()
    np.savetxt('eigvals_g0w0_pyscf_rescf_he_0089.txt', gw.mo_energy_gw[0,:,:].T)

          
if __name__ == "__main__": unittest.main()
