from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf, conv_yzx2xyz_c
from pyscf import gto, scf as scf_gto

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)
        
conv = conv_yzx2xyz_c(mol)

class KnowValues(unittest.TestCase):

  def test_dens_elec(self):
    """ Compute density in coordinate space with scf, integrate and compare with number of electrons """
    from timeit import default_timer as timer
    
    gto_mf = scf_gto.RKS(mol)
    gto_mf.kernel()
        
    sv = mf(mf=gto_mf, gto=mol)
    dm = sv.make_rdm1()
    grid = sv.build_3dgrid_pp(level=5)
    dens = sv.dens_elec(grid.coords, dm)
    nelec = np.einsum("is,i", dens, grid.weights)[0]

    self.assertAlmostEqual(nelec, sv.nelectron, 6)
    
      
if __name__ == "__main__": unittest.main()
