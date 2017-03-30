import numpy

#
#
#
class sbt_c():
  '''
  Spherical Bessel Transform by James Talman. Functions are given on logarithmic mesh
  See m_log_mesh
  Args:
    nr : integer, number of points on radial mesh
    rr : array of points in coordinate space
    kk : array of points in momentum space
    lmax : integer, maximal angular momentum necessary
    with_sqrt_pi_2 : if one, then transforms will be multiplied by sqrt(pi/2)
    fft_flags : ??
  Returns:
    a class preinitialized to perform the spherical Bessel Transform
  
  Examples:
    
  '''
  def __init__(self, rr=None, kk=None, lmax=12, with_sqrt_pi_2=0, fft_flags=None):
    assert(type(rr)==numpy.ndarray)
    assert(rr[0]>0.0)
    assert(type(kk)==numpy.ndarray)
    assert(kk[0]>0.0)
    self.nr = len(rr)
    assert(self.nr>1)
    self.lmax = lmax
    assert(self.lmax>-1)
    self.rr = rr
    self.kk = kk
    self.with_sqrt_pi_2 =with_sqrt_pi_2
    self.fft_flags = fft_flags
    self.nr2 = self.nr*2
    self.rr3 = rr**3
    self.kk3 = kk**3
    self.premult = numpy.zeros((self.nr2), dtype='float64')
    self.smallr  = numpy.zeros((self.nr), dtype='float64')
    self.postdiv = numpy.zeros((self.nr), dtype='float64')
    self.mult_table1 = numpy.zeros((self.lmax, self.nr), dtype='float64')
    self.mult_table2 = numpy.zeros((self.lmax, self.nr+1), dtype='float64')
    self.rmin = rr[0]
    self.kmin = kk[0]
    self.rhomin = numpy.log(self.rmin)
    self.kapmin = numpy.log(self.kmin)
    dr = numpy.log(rr[2]/rr[1])  
    dt = 2.0*numpy.pi/(self.nr2*dr) 
    
    self.temp1 = numpy.zeros((self.nr2), dtype='complex128')
    self.temp2 = numpy.zeros((self.nr2), dtype='complex128')
    self.temp1[0] = 1.0
    self.temp2 = numpy.fft.fft(self.temp1)
    xx = sum(numpy.real(self.temp2))
    if abs(self.nr2-xx)>1e-10 : raise SystemError('err: sbt_plan: problem with fftw sum(temp2):')

    factor = numpy.exp(dr)
    self.smallr[self.nr-1] = self.rr[0]/factor
    for i in range(self.nr-2,-1,-1): self.smallr[i] = self.smallr[i+1]/factor

    factor = numpy.exp(1.5*dr)
    self.premult[self.nr] = 1.0  
    for i in range(1,self.nr): self.premult[self.nr+i] = factor*self.premult[self.nr+i-1]

  #p%premult(nr) = 1.0D0/factor
  #do i = 2,nr  
     #p%premult(nr-i+1) = p%premult(nr-i+2)/factor
  #enddo

  #!   Obtain the values 1/k_i^1/5 in the array postdivide
  #p%postdiv(1) = 1.0D0
  #if(with_sqrt_pi_2) p%postdiv(1) = 1.0D0/sqrt(pi/2)

  #do i = 2,nr
     #p%postdiv(i) = p%postdiv(i-1)/factor 
  #enddo

  #!   construct the array of M_l(t) times the phase
  #do it = 1,nr
     #tt = (it-1)*dt               ! Define a t value
     #phi3 = (kappamin+rhomin)*tt  ! See Eq. (33)
     #rad = sqrt(10.5D0**2+tt**2)
     #phi = atan((2D0*tt)/21D0)
     #phi1 = -10D0*phi-log(rad)*tt+tt+sin(phi)/(12D0*rad)&
          #-sin(3D0*phi)/(360D0*rad**3)+sin(5D0*phi)/(1260D0*rad**5)&
          #-sin(7D0*phi)/(1680D0*rad**7)
     #do ix = 1,10
        #phi = 2*tt/(2D0*ix-1)
        #phi1 = phi1+atan((2D0*tt)/(2D0*ix-1))  ! see Eqs. (27) and (28)
     #enddo

     #if(tt>200d0) then
       #phi2 = -atan(1d0)
     #else  
       #phi2 = -atan(sinh(pi*tt/2)/cosh(pi*tt/2))  ! see Eq. (20)
     #endif  
     #phi = phi1+phi2+phi3
     #p%mult_table1(it,0) = sqrt(pi/2)*exp(ci*phi)/nr  ! Eq. (18)
     #if (it.eq.1) p%mult_table1(it,0) = 0.5D0*p%mult_table1(it,0)
     #phi = -phi2-atan(2D0*tt)
     #if(p%lmax>0)p%mult_table1(it,1) = exp(2.0D0*ci*phi)*p%mult_table1(it,0) ! See Eq. (21)
      #!    Apply Eq. (24)
     #do lk = 1,p%lmax-1
        #phi = -atan(2*tt/(2*lk+1))
        #p%mult_table1(it,lk+1) = exp(2.0D0*ci*phi)*p%mult_table1(it,lk-1)
     #enddo
  #enddo

  #!   make the initialization for the calculation at small k values
  #!   for 2N mesh values

  #allocate (j_ltable(nr2,0:p%lmax))
  #allocate(xj(0:p%lmax))

  #!   construct a table of j_l values

  #do i = 1,nr2
     #xx = exp(rhomin+kappamin+(i-1)*dr)  
     #call XJL(xx,p%lmax,xj)
     #do ll = 0,p%lmax
        #j_ltable(i,ll) = xj(ll)
     #enddo
  #enddo

  #do ll = 0,p%lmax
     #temp1(1:nr2) = j_ltable(1:nr2,ll)
     #call dfftw_execute(plan12)
     #p%mult_table2(1:nr+1,ll) = conjg(temp2(1:nr+1))/nr2
  #enddo
  #if(with_sqrt_pi_2) p%mult_table2 = p%mult_table2/sqrt(pi/2)

  #deallocate(j_ltable)
  #deallocate(xj)
#end subroutine !INITIALIZE
