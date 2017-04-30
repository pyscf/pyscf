
import numpy

#
#
#
def log_mesh(nr, rmin, rmax, kmax=None):
  """
  Initializes log grid in real and reciprocal (momentum) spaces.
  These grids are used in James Talman's subroutines. 
  """
  assert(type(nr)==int)

  rhomin=numpy.log(rmin)
  rhomax=numpy.log(rmax)
  kmax = 1.0/rmin/numpy.pi if kmax is None else kmax
  kapmin=numpy.log(kmax)-rhomax+rhomin

  rr=numpy.array(numpy.exp( numpy.linspace(rhomin, rhomax, nr), dtype='float64') )
  pp=numpy.array(rr*(numpy.exp(kapmin)/rr[0]), dtype='float64')

  return rr, pp
  
