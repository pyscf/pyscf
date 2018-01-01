from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

mol = gto.M( verbose = 1, atom = '''O 0 0 0; O 0 0 2.0''', basis = 'cc-pvdz', spin = 2, )
gto_mf_uhf = scf.UHF(mol)
e_tot = gto_mf_uhf.kernel()
jg,kg = gto_mf_uhf.get_jk()

class KnowValues(unittest.TestCase):

  def test_o2_gw_0087(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case GW procedure. """
    #print(__name__, dir(gto_mf_uhf))
    gw = gw_c(mf=gto_mf_uhf, gto=mol, verbosity=0, niter_max_ev=6)
    print(__name__, 'nfermi =', gw.nfermi)
    print(__name__, 'e_tot =', e_tot)
    self.assertEqual(gw.nspin, 2)
    
    gw.kernel_gw()
    print(gw.mo_energy)
    print(gw.mo_energy_gw)
    print(gw.ksn2f)
    #print(__name__, gw.nspin)
    
    #print(__name__)
    #print(gw.mo_occ)
    #print(gw.nelec)
    #print(gw.nelectron)
    #print(gw.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
