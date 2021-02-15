from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''Mn 0 0 0;''', basis = 'cc-pvdz', spin = 5, )
#gto_mf_rhf = scf.RHF(mol)
#gto_mf_rhf.kernel()
gto_mf_uhf = scf.UHF(mol)
gto_mf_uhf.kernel()
dm = gto_mf_uhf.make_rdm1()
#print(__name__, dm.shape, type(dm))
jg,kg = gto_mf_uhf.get_jk()
#print(jg.shape, kg.shape)
#print(__name__, (jg*dm).sum()/2.0)
#print(__name__, (kg*dm).sum()/2.0)
#print('vh0-vh1', (jg[0]-jg[1]).sum())

class KnowValues(unittest.TestCase):

  def test_mn_gw_0086(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case GW procedure. """
    #print(__name__, dir(gto_mf_uhf))
    gw = gw_c(mf=gto_mf_uhf, gto=mol, verbosity=0)
    self.assertEqual(gw.nspin, 2)
    
    #gw.kernel_gw()
    #print(__name__, gw.nspin)
    
    #print(__name__)
    #print(gw.mo_occ)
    #print(gw.nelec)
    #print(gw.nelectron)
    #print(gw.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
