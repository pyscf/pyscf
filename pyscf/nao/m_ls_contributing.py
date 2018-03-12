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
from numpy import sqrt

def ls_contributing(pb, sp12, ra12):
  """
  List of contributing centers 
  prod_basis_c : instance of the prod_basis_c containing parameters .ac_rcut_ratio and .ac_npc_max and .sv providing instance of system_vars_c which provides the coordinates, unit cell vectors, species etc. and .prod_log prividing information on the cutoffs for each specie.
  sp12 : a couple of species
  ra12 : a couple of coordinates, correspondingly
  """
  ra3 = 0.5*(ra12[0,:] + ra12[1,:])
  ia2dist = np.zeros(pb.sv.natoms)
  for ia,rvec in enumerate(pb.sv.atom2coord-ra3): ia2dist[ia] = sqrt((rvec**2).sum())
  return np.argsort(ia2dist)[0:pb.ac_npc_max]

