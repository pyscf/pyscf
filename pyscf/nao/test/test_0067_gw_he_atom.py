from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''He 0.0 0.0 0.0''', basis = 'aug-cc-pvtz', spin = 0, )
gto_mf = scf.UHF(mol)
e_tot = gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_0089_he_atom(self):
    """ Spin-resolved case GW procedure. """
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=0, niter_max_ev=16, kmat_timing=0.0, kmat_algo='sm0_sum')
    self.assertEqual(gw.nspin, 2)
    gw.kernel_gw()
    #if gw.kmat_timing is not None: print('gw.kmat_timing', gw.kmat_timing)
    #gw.report()
    np.savetxt('eigvals_gw_pyscf_he_0067.txt', gw.mo_energy_gw[0,:,:].T)
    ev_ref = np.loadtxt('eigvals_gw_pyscf_he_0067.txt-ref').T
    for n2e,n2r in zip(gw.mo_energy_gw[0], ev_ref):
      for e,r in zip(n2e,n2r): self.assertAlmostEqual(e, r)
          
if __name__ == "__main__": unittest.main()
