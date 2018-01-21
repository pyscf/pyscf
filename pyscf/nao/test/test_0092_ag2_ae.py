from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import nao, scf as scf_c
from pyscf.nao import prod_basis_c

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 0, atom = '''Ag 0.0 0.0 -0.3707;  Ag 0.0 0.0 0.3707''', basis = 'cc-pvdz-pp',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    print('gto_mf.mo_energy:', gto_mf.mo_energy)
    s = nao(mf=gto_mf, gto=mol, verbosity=0)
    print('s.norbs:', s.norbs)
    pb = prod_basis_c()
    pb.init_prod_basis_gto(s)

    
        
if __name__ == "__main__": unittest.main()
