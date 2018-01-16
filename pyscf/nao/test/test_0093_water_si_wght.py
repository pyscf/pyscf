from __future__ import print_function, division
import unittest

class KnowValues(unittest.TestCase):

  def test_water_si_wgth(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao import gw as gw_c
    from pyscf.nao import mf as mf_c
    from pyscf.nao.m_x_zip import detect_maxima
    from pyscf.nao.m_lorentzian import overlap, lorentzian
    import numpy as np
    import os
    from numpy import arange, einsum, array, linalg, savetxt, column_stack, conj
    from scipy.integrate import simps
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = gw_c(label='water', cd=dname, verbosity=1, nocc=8, nvrt=6, rescf=False, tol_ia=1e-9)
    #gw.kernel_gw()
    weps = 0.25
    wmax = 2*(mf.mo_energy[0,0,-1]-mf.mo_energy[0,0,0])
    ww = arange(0.0, wmax, weps/3.0)+1j*weps
    si0 = mf.si_c(ww)
    hk_inv = linalg.inv(mf.hkernel_den)
    print(si0.shape, hk_inv.shape)
    si0_dens = -einsum('wpq,pq->w', si0, hk_inv).imag
    savetxt('w2scr_int.txt', column_stack((ww.real, si0_dens)))
    wwmx = detect_maxima(ww, si0_dens)
    print(wwmx)
    oo = overlap(wwmx, weps)
    np.set_printoptions(linewidth=156)
    print(oo)
    
    wwll = np.zeros(len(wwmx), dtype=np.complex128)
    for i,wn in enumerate(wwmx):
      wwll[i] = simps(si0_dens*conj(lorentzian(ww.real, wn, weps)), ww.real)
    
    print( np.linalg.solve(oo, wwll) )
      
    
    
    

if __name__ == "__main__": unittest.main()
