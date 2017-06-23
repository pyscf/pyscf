from __future__ import print_function, division
import unittest


class KnowValues(unittest.TestCase):

  def test_get_vrtx_cc_apair(self):
    """ This is for poke into a buffer to transfer to the library """
    from pyscf.nao import ao_log_c
    import os
    from pyscf.nao.m_libnao import libnao
    from ctypes import POINTER, c_double, c_int64
    from numpy import zeros

    d = os.path.dirname(os.path.abspath(__file__))
    s2i = [siesta_ion_xml(d+'/'+f+'.ion.xml') for f in ['H','O']]
    
    libnao.vrtx_cc_apair.argtypes = (
      POINTER(c_int64),    # ninp
      POINTER(c_double),   # dinp(ninp)
      POINTER(c_int64),    # nout
      POINTER(c_double) )  # dout(nout)

    svn = ao_log_c().init_ao_log_ion(s2i).get_aoneo() # construct representation of system_vars_c
    self.assertAlmostEqual(svn.sum(), 60906.882217023987) 
    dat = zeros(1000000)
    #libnao.vrtx_cc_apair(c_int64(len(svn)), svn.ctypes.data_as(POINTER(c_double)),
    #  c_int64(len(dat)), dat.ctypes.data_as(POINTER(c_double)))
    
if __name__ == "__main__":
  unittest.main()
