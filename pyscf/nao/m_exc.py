# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import numpy as np
from pyscf.dft import libxc

#
#
#
def exc(sv, dm, xc_code, **kvargs):
  """
    Computes the exchange-correlation energy for a given density matrix
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
      xc_code : is a string must comply with pySCF's convention PZ  
        "LDA,PZ"
        "0.8*LDA+0.2*B88,PZ"
    Returns:
      exc x+c energy
  """
  grid = sv.build_3dgrid_pp(**kvargs)
  dens = sv.dens_elec(grid.coords, dm)
  exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, dens.T, spin=sv.nspin-1, deriv=0)
  nelec = np.dot(dens[:,0]*exc, grid.weights)
  return nelec
