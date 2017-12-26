from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 5, atom = '''O 0 0 0; O 0 0 2.0''', basis = 'cc-pvdz', spin = 0, )
#gto_mf_rhf = scf.RHF(mol)
#gto_mf_rhf.kernel()
gto_mf_uhf = scf.UHF(mol)
e_tot = gto_mf_uhf.kernel()
dm = gto_mf_uhf.make_rdm1()
gto_mf_uhf.analyze()
print(__name__, gto_mf_uhf.nelec, gto_mf_uhf.spin_square(), e_tot, dir(gto_mf_uhf))
print(gto_mf_uhf.get_occ())
jg,kg = gto_mf_uhf.get_jk()
print(jg.shape, kg.shape)
print(__name__, (jg*dm).sum()/2.0)
print(__name__, (kg*dm).sum()/2.0)
print('vh0-vh1', (jg[0]-jg[1]).sum())


class KnowValues(unittest.TestCase):

  def test_o2_gw_0087(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case GW procedure. """
    #print(__name__, dir(gto_mf_uhf))
    #gw = gw_c(mf=gto_mf_uhf, gto=mol, verbosity=1)
    #self.assertEqual(gw.nspin, 2)
    
    #gw.kernel_gw()
    #print(__name__, gw.nspin)
    
    #print(__name__)
    #print(gw.mo_occ)
    #print(gw.nelec)
    #print(gw.nelectron)
    #print(gw.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
