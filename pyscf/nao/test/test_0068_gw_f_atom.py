from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''F 0.0 0.0 0.0''', basis = 'aug-cc-pvdz', spin = 1, )
gto_mf = scf.UHF(mol)
e_tot = gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_0068_F_atom(self):
    """ Spin-resolved case GW procedure. """
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=0, niter_max_ev=16, rescf=True, kmat_algo='dp_vertex_loops_sm')
    self.assertEqual(gw.nspin, 2)
    gw.kernel_gw()
    #gw.report()
    np.savetxt('eigvals_gw_pyscf_f_0068.txt', gw.mo_energy_gw[0,:,:].T)
    ev_ref = np.loadtxt('eigvals_gw_pyscf_f_0068.txt-ref').T
    for n2e,n2r in zip(gw.mo_energy_gw[0], ev_ref):
      for e,r in zip(n2e,n2r): self.assertAlmostEqual(e, r)
          
if __name__ == "__main__": unittest.main()
