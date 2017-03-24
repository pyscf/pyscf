import numpy
import sys
import re
from pyscf.nao.m_color import color as bc
from pyscf.nao.m_siesta_xml import siesta_xml
from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
from pyscf.nao.m_siesta_hsx import siesta_hsx_c
from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
from pyscf.nao.m_sv_diag import sv_diag 

#
#
#
def get_orb2m(sv):
  orb2m = numpy.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom in range(sv.natoms):
    sp = sv.atom2sp[atom]
    nmult = sv.sp2nmult[sp]
    for mu in range(nmult):
      j = sv.sp_mu2j[sp,mu]
      for m in range(-j,j+1):
        orb2m[orb] = m
        orb = orb + 1
  return(orb2m)

#
#
#
def diag_check(self):
  nspin = self.nspin
  nkpnt = self.nkpoints
  ksn2e = self.xml_dict['ksn2e']
  ac = True
  for k in range(nkpnt):
    kvec = self.xml_dict["k2xyzw"][k,0:3]
    for spin in range(nspin):
      e,x = sv_diag(self, kvec=kvec, spin=spin)
      eref = ksn2e[k,spin,:]
      acks = numpy.allclose(eref,e,atol=1e-5,rtol=1e-4)
      ac = ac and acks
      if(not acks):
        aerr = sum(abs(eref-e))/len(e)
        print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
  return(ac)


#
#
#
class system_vars_c():
  def __init__(self, label='siesta', forcetype=-1):
    self.xml_dict = siesta_xml(label)
    self.wfsx = siesta_wfsx_c(label)
    self.hsx = siesta_hsx_c(label, forcetype)
    
    ##### The parameters as fields     
    self.sp2ion = []
    for strspecie in self.wfsx.sp2strspecie: self.sp2ion.append(siesta_ion_xml(strspecie+'.ion.xml'))
    nsp = len(self.sp2ion)
    self.sp2nmult = numpy.empty((nsp), dtype='int64', order='F') 
    self.sp2nmult[:] = list(len(self.sp2ion[sp]["orbital"]) for sp in range(nsp))
    nmultmx = max(self.sp2nmult)
    self.sp_mu2j = numpy.empty((nsp,nmultmx), dtype='int64', order='F') 
    self.sp_mu2j.fill(-999)
    for sp in range(nsp):
      nmu = self.sp2nmult[sp]
      for mu in range(nmu):
        self.sp_mu2j[sp,mu] = self.sp2ion[sp]["orbital"][mu]["l"]
    
    self.natoms = len(self.xml_dict['atom2sp'])
    self.norbs  = (self.xml_dict['ksn2e'].shape[2])
    self.nspin  = (self.xml_dict['ksn2e'].shape[1])
    self.nkpoints  = (self.xml_dict['ksn2e'].shape[0])

    strspecie2sp = {}
    for sp in range(len(self.wfsx.sp2strspecie)): strspecie2sp[self.wfsx.sp2strspecie[sp]] = sp
    
    self.atom2sp = numpy.empty((self.natoms), dtype='int64')
    for o in range(self.wfsx.norbs):
      atom = self.wfsx.orb2atm[o]
      strspecie = self.wfsx.orb2strspecie[o]
      self.atom2sp[atom-1] = strspecie2sp[strspecie]
    
    orb2m = get_orb2m(self)
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.X[:,:,n,s,k])

    #print(get_denmat(self).shape)    
    #print(self.sp2nmult, type(self.sp2nmult))
    #print(self.sp_mu2j, type(self.sp_mu2j))
    #print(self.sp2ion[0].keys())
