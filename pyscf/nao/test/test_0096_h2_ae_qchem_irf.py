from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf, gw as gto_gw
from pyscf.nao.qchem_inter_rf import qchem_inter_rf
from pyscf.nao import tddft_iter

mol = gto.M( verbose = 1,
    atom = '''
        H    0.0   2.0      0.0
        H   -1.0   0.0      0.0
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pVDZ',)

class KnowValues(unittest.TestCase):
    
  def test_qchem_irf(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    gto_hf = scf.RHF(mol)
    gto_hf.kernel()
    print(gto_hf.mo_energy)
    
    qrf = qchem_inter_rf(mf=gto_hf, gto=mol, pb_algorithm='fp', verbosity=1)
    td = tddft_iter(mf=gto_hf, gto=mol, pb_algorithm='fp', verbosity=1, xc_code='RPA', tddft_iter_tol=1.1e-6)
    
    ww = np.arange(0.0, 1.0, 0.1)+1j*0.2
    rf1_qc = qrf.inter_rf(ww)

    rf1_mv  = np.zeros_like(rf1_qc)
    vec = np.zeros((td.nprod), dtype=td.dtypeComplex)
    for iw,w in enumerate(ww):
      for mu in range(td.nprod):
        vec[:] = 0.0; vec[mu] = 1.0
        veff = td.comp_veff(vec, w)
        rf1_mv[iw, mu,:] = td.apply_rf0(veff, w)

    print('abs(rf1_qc-rf1_mv).sum()', abs(rf1_qc-rf1_mv).sum())
    for w,rf1,rf2 in zip(ww,rf1_qc,rf1_mv):
      print(w, rf1[1,0:1], rf2[1,0:1])
    
    
if __name__ == "__main__": unittest.main()
