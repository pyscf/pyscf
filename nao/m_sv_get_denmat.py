import numpy

#
#
#
def sv_get_denmat(sv, mattype='hamiltonian', prec='64', kvec=[0.0,0.0,0.0], spin=0):

  mat = None
  
  if(sv.hsx.is_gamma):

    mat = numpy.empty((sv.norbs,sv.norbs), dtype='float'+prec)
    mtl = mattype.lower()  
    if(mtl=='hamiltonian'):
      mat = sv.hsx.spin2h4_csr[spin].todense()
    elif(mtl=='overlap'):
      mat = sv.hsx.s4_csr.todense()
    else: 
      raise SystemError('!mattype')

  else:
    mat = numpy.empty((sv.norbs,sv.norbs), dtype='complex'+prec)
    raise SystemError('!impl')
    
  return mat
