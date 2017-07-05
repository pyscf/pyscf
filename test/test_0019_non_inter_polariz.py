from __future__ import print_function, division
import os,unittest
import numpy as np
from pyscf.nao import system_vars_c, prod_basis_c, tddft_iter_c

dname = os.path.dirname(os.path.abspath(__file__))
sv = system_vars_c().init_siesta_xml(label='water', chdir=dname)
pb = prod_basis_c().init_pb_pp_libnao_apair(sv)
pb.init_prod_basis_pp()

class KnowValues(unittest.TestCase):

  def test_non_inter_polariz(self):
    """ This is iterative TDDFT with SIESTA starting point """
    td = tddft_iter_c(pb.sv, pb, tddft_iter_broadening=1e-2)
    omegas = np.linspace(0.0,2.0,500)
    pxx = np.zeros_like(omegas)
    vext = np.transpose(td.moms1)
    Hartree2eV = 27.2114
    for iomega,omega in enumerate(omegas):
      dn0 = td.apply_rf0(vext[0,:], omega)
      pxx[iomega] = -np.dot(dn0, vext[0,:]).imag
    
    data = np.array([omegas*Hartree2eV, pxx])
    np.savetxt('water.tddft_iter.omega.pxx.txt', data.T, fmt=['%f','%f'])


if __name__ == "__main__":
  unittest.main()
