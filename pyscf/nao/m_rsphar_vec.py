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

import numpy as np
from pyscf.symm import sph

#
def rsphar_vec(rvecs, lmax):
    """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      rvecs : A list of Cartesian coordinates defining the theta and phi angles for spherical harmonic
      lmax  : Integer, maximal angular momentum
    Result:
      2-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
    """
    assert lmax>-1
    ylm = sph.real_sph_vec(rvecs, lmax)
    res = np.vstack(ylm).T.copy('C')
    return res

