from __future__ import division, print_function
from pyscf.nao.m_ao_matelem import build_3dgrid
from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
from pyscf.nao.m_dens_libnao import dens_libnao as dens
from numpy import einsum

#
#
#
def xc_scalar_ni(me, sp1,R1, sp2,R2, **kvargs):
  from pyscf.dft.libxc import eval_xc
  """
    Computes overlap for an atom pair. The atom pair is given by a pair of species indices
    and the coordinates of the atoms.
    Args: 
      sp1,sp2 : specie indices, and
      R1,R2 :   respective coordinates in Bohr, atomic units
    Result:
      matrix of orbital overlaps
    The procedure uses the numerical integration in coordinate space.
  """
    
  grids = build_3dgrid(me, sp1, R1, sp2, R2, **kvargs)
  rho = dens(grids.coords, me.sv.nspin)
  ao1 = ao_eval(me.ao1, R1, sp1, grids.coords)
  ao1 = ao1 * grids.weights * rho[:,0]
  ao2 = ao_eval(me.ao2, R2, sp2, grids.coords)
  overlaps = einsum("ij,kj->ik", ao1, ao2)

  return overlaps
