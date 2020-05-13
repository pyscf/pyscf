from __future__ import print_function, division
from numpy import array, argmax
from pyscf.nao import tddft_iter


class tddft_iter_x_zip(tddft_iter):
  """ Iterative TDDFT with a high-energy part of the KS eigenvectors compressed """

  def __init__(self, **kw):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    
    tddft_iter.__init__(self, **kw)
    self.x_zip = kw['x_zip'] if 'x_zip' in kw else False
    self.x_zip_eps = kw['x_zip_eps'] if 'x_zip_eps' in kw else 0.05
    self.x_zip_emax = kw['x_zip_emax'] if 'x_zip_emax' in kw else 0.25

    if self.x_zip: # redefine the eigenvectors
      sm2e,sma2x = self.build_x_zip()
      if self.verbosity>0: 
        print(__name__, 'self.mo_energy.shape =', self.mo_energy.shape)
        print(__name__, 'sm2e.shape =', sm2e.shape)
      self.ksn2e = array([sm2e])
      ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
      for s,n2fd in enumerate(ksn2fd[0]):
        if not all(n2fd>self.nfermi_tol): continue
        print(self.telec, s, nfermi_tol, n2fd)
        raise RuntimeError(__name__, 'telec is too high?')
        
      self.ksn2f = (3-self.nspin)*ksn2fd
      self.nfermi = array([argmax(ksn2fd[0,s,:]<self.nfermi_tol) for s in range(self.nspin)], dtype=int)
      self.vstart = array([argmax(1.0-ksn2fd[0,s,:]>=self.nfermi_tol) for s in range(self.nspin)], dtype=int)
      self.xocc = [ma2x[:nfermi,:] for ma2x,nfermi in zip(sma2x,self.nfermi)]
      self.xvrt = [ma2x[vstart:,:] for ma2x,vstart in zip(sma2x,self.vstart)]

  def build_x_zip(self):
    """ define compressed eigenvectors """
    from pyscf.nao.m_x_zip import x_zip
    sm2e = []
    sma2x = []
    for n2e,na2x in zip(self.mo_energy[0], self.mo_coeff[0,:,:,:,0]):
      vst, i2w,i2dos, m2e, ma2x = x_zip(n2e, na2x, eps=self.x_zip_eps, emax=self.x_zip_emax)
      sm2e.append(m2e)
      sma2x.append(ma2x)
    sm2e = array(sm2e)    
    return sm2e, sma2x
