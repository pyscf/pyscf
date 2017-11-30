from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_rf0_ref(self):
    """ This is GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol)
    ww = [0.0+1j*4.0, 1.0+1j*0.1, -2.0-1j*0.1]

    rf0_fm = gw.rf0_cmplx_vertex_ac(ww)
    rf0_mv  = np.zeros_like(rf0_fm)
    vec = np.zeros((gw.nprod), dtype=gw.dtypeComplex)
    for iw,w in enumerate(ww):
      for mu in range(gw.nprod):
        vec[:] = 0.0; vec[mu] = 1.0
        rf0_mv[iw, mu,:] = gw.apply_rf0(vec, w)

    #print(rf0_fm.shape, rf0_mv.shape)
    #print('abs(rf0_fm-rf0_mv)', abs(rf0_fm-rf0_mv).sum()/rf0_fm.size)
    #print(abs(rf0_fm[0,:,:]-rf0_mv[0,:,:]).sum())
    #print(rf0_fm[0,:,:])
    self.assertTrue(abs(rf0_fm-rf0_mv).sum()/rf0_fm.size<1e-15)

if __name__ == "__main__": unittest.main()
