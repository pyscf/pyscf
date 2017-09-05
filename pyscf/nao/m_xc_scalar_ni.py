from __future__ import division, print_function
from pyscf.nao.m_ao_matelem import build_3dgrid
from pyscf.nao.m_dens_libnao import dens_libnao
from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
from numpy import einsum
from pyscf.dft import libxc

#
#
#
def xc_scalar_ni(me, sp1,R1, sp2,R2, xc_code, deriv, **kvargs):
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
  rho = dens_libnao(grids.coords, me.sv.nspin)
  exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho.T, spin=me.sv.nspin-1, deriv=deriv)
  ao1 = ao_eval(me.ao1, R1, sp1, grids.coords)
  if deriv==1 :
    #print(' vxc[0].shape ', vxc[0].shape)
    ao1 = ao1 * grids.weights * vxc[0]
  elif deriv==2:
    ao1 = ao1 * grids.weights * fxc[0]
  else:
    print(' deriv ', deriv)
    raise RuntimeError('!deriv!')

  ao2 = ao_eval(me.ao2, R2, sp2, grids.coords)
  overlaps = einsum("ij,kj->ik", ao1, ao2)

  return overlaps
