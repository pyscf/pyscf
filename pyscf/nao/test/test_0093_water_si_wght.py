from __future__ import print_function, division
import unittest

class KnowValues(unittest.TestCase):

  def test_water_si_wgth(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao import gw as gw_c
    from pyscf.nao import mf as mf_c
    from pyscf.nao.m_x_zip import detect_maxima
    from pyscf.nao.m_lorentzian import overlap, lorentzian, overlap_imag, overlap_real
    from pyscf.nao.m_sf2f_rf import sf2f_rf
    import numpy as np
    import os
    from numpy import arange, einsum, array, linalg, savetxt, column_stack, conj
    from scipy.integrate import simps
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = gw_c(label='water', cd=dname, verbosity=0, nocc=8, nvrt=6, rescf=False, tol_ia=1e-9)
    #gw.kernel_gw()
    weps = 0.3
    wmax = 1.1*(mf.mo_energy[0,0,-1]-mf.mo_energy[0,0,0])
    ww = arange(0.0, wmax, weps/3.0)+1j*weps
    si0 = mf.si_c(ww)
    hk_inv = linalg.inv(mf.hkernel_den)
    print(__name__, si0.shape, hk_inv.shape)
    si0_dens = -einsum('wpq,pq->w', si0, hk_inv).imag
    si0_dens_re = einsum('wpq,pq->w', si0, hk_inv).real
    savetxt('w2scr_int.txt', column_stack((ww.real, si0_dens)))
    savetxt('w2scr_int_re.txt', column_stack((ww.real, si0_dens_re)))
    wwmx = list(detect_maxima(ww, si0_dens))
    print('nmax', len(wwmx))
    
    dww = ww[1].real-ww[0].real 
    # Method 1
    sf_si0 = np.zeros((len(wwmx), mf.nprod, mf.nprod))    
    for j,wmx in enumerate(wwmx): sf_si0[j] = -si0[np.argmin(abs(ww.real - wmx))].imag/np.pi

    # Method 2, using imaginary part
    sf_si0 = np.zeros((len(wwmx), mf.nprod, mf.nprod))
    for j,wmx in enumerate(wwmx):
      for i,fw in enumerate(ww.real): 
        sf_si0[j] += si0[i].imag*dww*lorentzian(fw, wmx, weps).imag

    loi = overlap_imag(wwmx, weps)
    iloi = np.linalg.inv(loi)
    sf_si0 = einsum('fg,gab->fab', iloi,sf_si0) 
    
    ## Method 3, using real part
    #re_si0 = np.zeros((len(wwmx), mf.nprod, mf.nprod))
    #for j,wmx in enumerate(wwmx):
      #for i,fw in enumerate(ww.real): 
        #re_si0[j] += si0[i].real*dww*lorentzian(fw, wmx, weps).real

    #lor = overlap_real(wwmx, weps)
    #ilor = np.linalg.inv(lor)
    #sf_si0 = einsum('fg,gab->fab', ilor,re_si0) 
    
    ivec = 0
    for i,sf in enumerate(sf_si0):
      ee,xx = np.linalg.eigh(sf)
      sf_si0[i] = 0.0;
      for e,x in zip(ee,xx.T):
        if e>0.01: 
          sf_si0[i] += np.outer(x*e, x)
          ivec += 1
    print('nvecs', ivec)

    si0_recon = sf2f_rf(ww.real, weps, wwmx, sf_si0)
    si0_dens_recon = -einsum('wpq,pq->w', si0_recon, hk_inv).imag
    si0_dens_recon_re = einsum('wpq,pq->w', si0_recon, hk_inv).real
    savetxt('w2scr_int_recon.txt', column_stack((ww.real, si0_dens_recon)))
    savetxt('w2scr_int_recon_re.txt', column_stack((ww.real, si0_dens_recon_re)))

    

if __name__ == "__main__": unittest.main()
