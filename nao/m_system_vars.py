import numpy
import sys
import re
from pyscf.nao.m_siesta_xml import siesta_xml
from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml

#
#
#
class system_vars_c():
  def __init__(self, label):
    self.xml_dict = siesta_xml(label)
    self.wfsx = siesta_wfsx_c(label)
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

    #print(self.sp2nmult, type(self.sp2nmult))
    #print(self.sp_mu2j, type(self.sp_mu2j))
    #print(self.sp2ion[0].keys())
    
