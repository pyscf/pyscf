from __future__ import print_function, division
from numpy import sqrt, array


def ls_part_centers(sv, ia1, ia2, ac_rcut_coeff=1.0):
  """
    For a given atom pair, defined with it's atom indices,
    list of the atoms within a radii: list of participating centers.
    The subroutine goes over radial orbitals of atoms ia1 and ia2
    computes the radii of products, the positions of centers of these
    products and then goes through atoms choosing these which overlap
    with the product.
  """
  assert(ac_rcut_coeff>0.0 and ac_rcut_coeff<=1.0)
  ls = []
  rv1, rv2 = sv.atom2coord[ia1], sv.atom2coord[ia2] 
  d = sqrt(sum((rv1-rv2)**2))
  dp2 = d**2
  
  rcut1 = sv.ao_log.sp2rcut[sv.atom2sp[ia1]]
  rcut2 = sv.ao_log.sp2rcut[sv.atom2sp[ia2]]
  #for mu1,rcut1 in enumerate(sv.ao_log.sp_mu2rcut[sv.atom2sp[ia1]]):
    #for mu2,rcut2 in enumerate(sv.ao_log.sp_mu2rcut[sv.atom2sp[ia2]]):
      #if rcut1+rcut2<d: continue
  dif2 = rcut2**2-rcut1**2+dp2
  a = dif2/2/dp2
  rmax2_nocond = rcut1**2 - dp2*a**2
  rmid_nocond = rv1*(a)+rv2*(1-a)

  a = a if a<1.0 else 1.0
  a = a if a>0.0 else 0.0
  rmid = rv1*(a)+rv2*(1-a)
  rmax2 = rcut1**2 - dp2*a**2
  rmid = rv1*(a)+rv2*(1-a)

  if rmax2<=0 :
    print('skip', d, rcut1, rcut2, a, rmax2, sqrt(sum((rmid-rv1)**2)), d*(1-a), sqrt(sum((rmid-rv2)**2)), d*a) 
    #continue

  print(rcut1, rcut2, d, a, rmid_nocond, sqrt(rmax2_nocond))

  if a==0.0 or a==1.0 : 
    rmax2 = min(rcut1,rcut2)**2
  
  print(rcut1, rcut2, d, a, rmid, sqrt(rmax2))

  for ia3, [ra,sp] in enumerate(zip(sv.atom2coord, sv.atom2sp)):
    rcut3 = sv.ao_log.sp2rcut[sv.atom2sp[ia3]]
    dist123 = ((ra-rmid)**2).sum()
    #print(d, rcut1, rcut2, rmid, sqrt(rmax2), sqrt(dist123), ia, rmax2>dist123)
    if (sqrt(rmax2)+rcut3)**2*ac_rcut_coeff>dist123: ls.append(ia3)

  return sorted(list(set(ls)))

#
#
#
def get_rmid_rcut(rv1,rcut1, rv2,rcut2):
  """
    For a given atom pair delivers coordinates of a center and 
    radius of product.
  """
  d = sqrt(sum((rv1-rv2)**2))
  dp2 = d**2
  dif2 = rcut2**2-rcut1**2+dp2
  a = dif2/2/dp2
  rmax2_nocond = rcut1**2 - dp2*a**2
  rmid_nocond = rv1*(a)+rv2*(1-a)

  a = a if a<1.0 else 1.0
  a = a if a>0.0 else 0.0
  rmid = rv1*(a)+rv2*(1-a)
  rmax2 = min(rcut1**2, rcut2**2)
  rmid = rv1*(a)+rv2*(1-a)
  return rmid, sqrt(rmax2)

#
#
#
def is_overlapping(rv1,rcut1, rv2,rcut2, rv3,rcut3):
  """ For a given atom pair (1,2) and a center 3 tell whether function at center overlaps with the product."""
  cond = sum((rv2-rv3)**2)<(rcut2+rcut3)**2 and sum((rv1-rv3)**2)<(rcut1+rcut3)**2
  return cond
  
#
#
#
if __name__ == '__main__':
  from pyscf import gto
  from pyscf.nao import system_vars_c
  import matplotlib.pyplot as plt
  import pylab

  rv1,rcut1 = array([3.0, 0.0, 0.0]), 3.0
  rv2,rcut2 = array([0.0, 0.0, 0.0]), 5.0
  rv3,rcut3 = array([-0.5, -3.0, 0.0]), 2.0
    
  axes = pylab.axes()
  axes.add_patch(pylab.Circle((rv1[0], rv1[1]), radius=rcut1, alpha=.33, color='r', edgecolor='y', lw=3))
  axes.add_patch(pylab.Circle((rv2[0], rv2[1]), radius=rcut2, alpha=.33, color='r', edgecolor='y', lw=3))
  axes.add_patch(pylab.Circle((rv3[0], rv3[1]), radius=rcut3, alpha=.33, color='blue'))
  pylab.axis('scaled')

  print(rv1, rcut1)
  print(rv2, rcut2)
  print(rv3, rcut3)
  print(is_overlapping(rv1,rcut1, rv2,rcut2, rv3,rcut3))

  pylab.show()

  
  #for coo,sp in zip(sv.atom2coord, sv.atom2sp):
  #  axes.add_patch(pylab.Circle((coo[2], coo[1]), radius=sv.ao_log.sp2rcut[sp], alpha=.2))
  #pylab.axis('scaled')
  #pylab.show()

