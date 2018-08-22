from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''H 0.0 0.0 0.0''', basis = 'aug-cc-pvtz', spin = 1, )
gto_mf = scf.RHF(mol)
e_tot = gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_0090_h_atom(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from io import StringIO
    """ Spin-resolved case GW procedure. """
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=2, niter_max_ev=16, kmat_algo='dp_vertex_loops_sm')
    self.assertEqual(gw.nspin, 1)
    gw.kernel_gw()
    gw.report()
    np.savetxt('eigvals_g0w0_pyscf_rescf_h_0090.txt', gw.mo_energy_gw[0,:,:].T)

          
if __name__ == "__main__": unittest.main()
