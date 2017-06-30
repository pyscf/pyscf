from __future__ import print_function, division
import unittest


class KnowValues(unittest.TestCase):

  def test_get_vrtx_cc_apair(self):
    """ This is for poke into a buffer to transfer to the library """
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao import ao_log_c
    import os
    from ctypes import POINTER, c_double, c_int64
    from numpy import zeros

    d = os.path.dirname(os.path.abspath(__file__))
    s2i = [siesta_ion_xml(d+'/'+f+'.ion.xml') for f in ['H','O']]
    svn = ao_log_c().init_ao_log_ion(s2i).get_aoneo() # construct a representation of system_vars
    self.assertAlmostEqual(svn.sum(), 60906.882217023987) 
    
    
if __name__ == "__main__":
  unittest.main()
