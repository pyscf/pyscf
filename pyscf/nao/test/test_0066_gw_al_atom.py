from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''Al 0.0 0.0 0.0''', basis = 'cc-pvdz', spin = 1, )
gto_mf = scf.UHF(mol)
e_tot = gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_0066_al_atom(self):
    """ Spin-resolved case GW procedure. """
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=0, niter_max_ev=16, nocc=3, nvrt=3)
    self.assertEqual(gw.nspin, 2)
    gw.kernel_gw()
    #gw.report()
    np.savetxt('eigvals_g0w0_pyscf_rescf_al_0066.txt', gw.mo_energy_gw[0,:,:].T)
    #ev_ref = np.loadtxt('eigvals_g0w0_pyscf_rescf_al_0066.txt-ref').T
    #for n2e,n2r in zip(gw.mo_energy_gw[0], ev_ref):
    #  for e,r in zip(n2e,n2r): self.assertAlmostEqual(e, r)
          
if __name__ == "__main__": unittest.main()
