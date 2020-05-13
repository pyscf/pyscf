from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw

class KnowValues(unittest.TestCase):

  def test_gw_h2_ae_spin_rf0_speed(self):
    """ This is GW """
    mol = gto.M( verbose = 1, 
      atom = '''H 0 0 0;  H 0 0 0.5; H 0 0 1.0; H 0 0 1.5; H 0 0 2.0; H 0 0 2.5;''', 
      basis = 'cc-pvdz', spin=0)
    #mol = gto.M( verbose = 0, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)
    gto_mf = scf.UHF(mol)
    etot = gto_mf.kernel()
    #print(__name__, 'etot', etot)
    #print('gto_mf.mo_energy:', gto_mf.mo_energy)
    b = gw(mf=gto_mf, gto=mol, verbosity=0, nvrt=4)
    ww = np.arange(0.0, 1.0, 0.1)+1j*0.2
    rf0 = b.rf0(ww)
    rf0_ref = b.rf0_cmplx_ref(ww)
    #print(__name__, len(b.x), '|diff|', abs(rf0_ref-rf0).sum()/rf0.size)
    self.assertTrue(abs(rf0_ref-rf0).sum()/rf0.size<1e-12)
    
        
if __name__ == "__main__": unittest.main()
