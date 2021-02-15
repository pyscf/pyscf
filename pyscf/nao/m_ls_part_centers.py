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
from numpy import sqrt


def ls_part_centers(sv, ia1, ia2, ac_rcut_ratio=1.0):
  """
    For a given atom pair, defined with it's atom indices,
    list of the atoms within a radii: list of participating centers.
    The subroutine goes over radial orbitals of atoms ia1 and ia2
    computes the radii of products, the positions of centers of these
    products and then goes through atoms choosing these which overlap
    with the product.
  """
  assert(ac_rcut_ratio>=0.0) 
  st = set()
  rv1, rv2 = sv.atom2coord[ia1], sv.atom2coord[ia2] 
  d = sqrt(sum((rv1-rv2)**2))
  dp2 = d**2

  for mu1,rcut1 in enumerate(sv.ao_log.sp_mu2rcut[sv.atom2sp[ia1]]):
    for mu2,rcut2 in enumerate(sv.ao_log.sp_mu2rcut[sv.atom2sp[ia2]]):
      if rcut1+rcut2<d: continue
      for ia3, [rv3,sp] in enumerate(zip(sv.atom2coord, sv.atom2sp)):
        rcut3 = sv.ao_log.sp2rcut[sv.atom2sp[ia3]]
        if is_overlapping(rv1,rcut1, rv2,rcut2, rv3,rcut3, ac_rcut_ratio): st.add(ia3)

  return sorted(list(st))

#
#
#
def is_overlapping(rv1,rcut1, rv2,rcut2, rv3,rcut3, ac_rcut_ratio=1.0):
  """ For a given atom pair (1,2) and a center 3 tell whether function at center overlaps with the product."""
  cond12 = sum((rv1-rv2)**2)<(rcut1+rcut2)**2
  cond = sum((rv2-rv3)**2)<ac_rcut_ratio**2*(rcut2+rcut3)**2 and sum((rv1-rv3)**2)<ac_rcut_ratio**2*(rcut1+rcut3)**2
  return cond and cond12
  
#
#
#
if __name__ == '__main__':
  from pyscf import gto
  from pyscf.nao import system_vars_c
  from numpy import array

  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; H 0 0 3; H 0 0 4', unit='angstrom') 
  sv = system_vars_c(gto=mol)
  print( ls_part_centers(sv, 0, 1, ac_rcut_ratio=0.7) )

  
  visualize_3c = False
  if visualize_3c:
    import matplotlib.pyplot as plt
    import pylab

    rv1,rcut1 = array([0.0,  0.0, 0.0]), 3.0
    rv2,rcut2 = array([5.0,  0.0, 0.0]), 3.0
    rv3,rcut3 = array([2.5,  0.0, 0.0]), 2.0
    
    axes = pylab.axes()
    axes.add_patch(pylab.Circle((rv1[0], rv1[1]), radius=rcut1, alpha=.33, color='r', edgecolor='y', lw=3))
    axes.add_patch(pylab.Circle((rv2[0], rv2[1]), radius=rcut2, alpha=.33, color='r', edgecolor='y', lw=3))
    axes.add_patch(pylab.Circle((rv3[0], rv3[1]), radius=rcut3, alpha=.33, color='blue'))
    pylab.axis('scaled')

    print(rv1, rcut1)
    print(rv2, rcut2)
    print(rv3, rcut3)
    print(is_overlapping(rv1,rcut1, rv2,rcut2, rv3,rcut3, 0.75))
    pylab.show()

  
  #for coo,sp in zip(sv.atom2coord, sv.atom2sp):
  #  axes.add_patch(pylab.Circle((coo[2], coo[1]), radius=sv.ao_log.sp2rcut[sp], alpha=.2))
  #pylab.axis('scaled')
  #pylab.show()

