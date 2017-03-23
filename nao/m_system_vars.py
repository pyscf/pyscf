import numpy
import sys
import re
from pyscf.nao.m_siesta_xml import siesta_xml 

#
#
#
class system_vars_c():
  def __init__(self, label):
    self.atom2coord, self.atom2sp, self.sp2elem = siesta_xml(label)
