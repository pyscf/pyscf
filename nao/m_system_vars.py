import numpy
import sys
import re
from pyscf.nao.m_siesta_xml import siesta_xml 

#
#
#
class system_vars_c():
  def __init__(self, label):
    self.xml_dict = siesta_xml(label)
