from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import nao, scf as scf_c
from pyscf.nao import prod_basis

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M(verbose=0, atom='''Ag 0 0 -0.3707; Ag 0 0 0.3707''', basis = 'cc-pvdz-pp',)
    gto_mf = scf.RHF(mol)#.density_fit()
    gto_mf.kernel()
    #print('gto_mf.mo_energy:', gto_mf.mo_energy)
    s = nao(mf=gto_mf, gto=mol, verbosity=0)
    oref = s.overlap_coo().toarray()
    #print('s.norbs:', s.norbs, oref.sum())

    pb = prod_basis(nao=s, algorithm='fp')
    pab2v = pb.get_ac_vertex_array()
    mom0,mom1=pb.comp_moments()
    orec = np.einsum('p,pab->ab', mom0, pab2v)
    self.assertTrue(np.allclose(orec,oref, atol=1e-3), \
      "{} {}".format(abs(orec-oref).sum()/oref.size, np.amax(abs(orec-oref))))

if __name__ == "__main__": unittest.main()
