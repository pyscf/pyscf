from __future__ import print_function, division
import unittest, sys, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 0, atom = '''H 0.0 0.0 0.0''', basis = 'aug-cc-pvdz', spin = 1, )
gto_mf = scf.RHF(mol)
e_tot = gto_mf.kernel()

#gto_rdm = gto_mf.make_rdm1()
#print(gto_rdm.shape)
#print((gto_mf.get_hcore()*gto_rdm).sum())
#print((gto_mf.get_j()*gto_rdm).sum())
#print((gto_mf.get_k()*gto_rdm).sum())
#mat = gto_mf.get_hcore()+gto_mf.get_j()-0.5*gto_mf.get_k()
#print(gto_mf.get_hcore().shape)
#print(gto_mf.get_j().shape)
#print(gto_mf.get_k().shape)
#print(gto_mf.mo_coeff.shape)
#x0 = gto_mf.mo_coeff[0,:]
#gto_expval = np.einsum('a,sab,b->s', x0, mat, x0)
#print(__name__, gto_expval)
#sys.exit()

class KnowValues(unittest.TestCase):

  def test_0090_h_atom(self):
    """ Spin-resolved case GW procedure. """
    return
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=2, niter_max_ev=16, kmat_algo='dp_vertex_loops_sm')
    #nao_rdm = gw.make_rdm1()
    #print(__name__, (gw.get_hcore()*nao_rdm).sum())
    #print((gw.get_j()*nao_rdm).sum())
    #print((gw.get_k()*nao_rdm).sum())
    
    self.assertEqual(gw.nspin, 1)
    gw.kernel_gw()
    gw.report()
    np.savetxt('eigvals_g0w0_pyscf_rescf_h_0090.txt', gw.mo_energy_gw[0,:,:].T)

          
if __name__ == "__main__": unittest.main()
