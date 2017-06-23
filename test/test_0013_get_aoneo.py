from __future__ import print_function, division
import unittest


class KnowValues(unittest.TestCase):

  def test_get_aoneo(self):
    """ This is for poke into a buffer to transfer to the library """
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao import ao_log_c
    import os
    from pyscf.nao.m_libnao import libnao
    from ctypes import POINTER, c_double, c_int64

    d = os.path.dirname(os.path.abspath(__file__))
    s2i = [siesta_ion_xml(d+'/'+f+'.ion.xml') for f in ['H','O']]
    svn = ao_log_c().init_ao_log_ion(s2i).get_aoneo()
    self.assertAlmostEqual(svn.sum(), 60906.882217023987)
    
    libnao.test_sv_get.argtypes = (
      POINTER(c_int64),    # n
      POINTER(c_double))     # svn(n)
  
    libnao.test_sv_get(c_int64(len(svn)), svn.ctypes.data_as(POINTER(c_double)))
    
    
if __name__ == "__main__":
  unittest.main()
