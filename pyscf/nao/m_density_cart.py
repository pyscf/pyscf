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

#
#
#
def density_cart(crds):
  """
    Compute the values of atomic orbitals on given grid points
    Args:
      sv     : instance of system_vars_c class
      crds   : vector where the atomic orbitals from "ao" are centered
      sab2dm : density matrix
    Returns:
      res[ncoord] : array of density
  """
  
  return np.zeros_like(crds)
