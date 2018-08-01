from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, 
  atom = '''Be 0.0 0.0 0.269654; H 0.0 0.0 -1.078616''', 
  basis = 'cc-pvdz', spin = 1, )
  
gto_mf_uhf = scf.UHF(mol)
e_tot = gto_mf_uhf.kernel()

class KnowValues(unittest.TestCase):

  def test_beh_gw_0088(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case GW procedure. """
    #print(__name__, dir(gto_mf_uhf))
    gw = gw_c(mf=gto_mf_uhf, gto=mol, verbosity=2, niter_max_ev=8, pb_algorithm='pp')
    print(__name__, 'nfermi =', gw.nfermi)
    print(__name__, 'e_tot =', e_tot)
    self.assertEqual(gw.nspin, 2)
    
    gw.kernel_gw()
    print(gw.mo_energy*27.2114)
    print(gw.mo_energy_gw*27.2114)
    print(gw.ksn2f)
    #print(__name__, gw.nspin)
    
    #print(__name__)
    #print(gw.mo_occ)
    #print(gw.nelec)
    #print(gw.nelectron)
    #print(gw.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
