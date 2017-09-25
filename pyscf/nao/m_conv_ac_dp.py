from __future__ import print_function, division
import numpy as np

class conv_ac_dp_c():

  def __init__(self, pb, dtype=np.float64):
    """ Conversion from atom-centered to dominant product basis and back, for vectors and tensors """
    self.pb = pb
    
