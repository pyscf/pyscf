from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c
from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
import os

mol = gto.M( verbose = 0, atom = '''Cl 0.0, 0.0, 0.514172 ; O 0.0, 0.0, -1.092615''',basis = os.path.abspath('../basis/cc-pvqz-l2.dat'), spin=1, charge=0)
gto_mf_UHF = scf.UHF(mol)
e_tot = gto_mf_UHF.kernel()



class KnowValues(unittest.TestCase):

  def test_clo_gw_0089(self):
    """ Spin-resolved case GW procedure. By using frozen_core convergence has been reached"""
    gw = gw_c(mf=gto_mf_UHF, gto=mol, verbosity=1, niter_max_ev=20, frozen_core=30) #Frozen core is defined by True, False or a number(N) which corrects state index in a range btween N-fermi and N+fermi   
    gw.kernel_gw()
    gw.report()
 

    self.assertEqual(gw.nspin, 2)
    self.assertAlmostEqual(gw.mo_energy_gw[0,0,2], -10.580537488540639)
    sf = gw.get_snmw2sf()
    self.assertEqual(len(sf), 2)                                    #without fz 2
    self.assertEqual(len(sf[0]), 18)                                #without fz 12
    self.assertEqual(sf[0].shape, (18, 68, 32))                     #without fz (12,68,32)
if __name__ == "__main__": unittest.main()
