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

from __future__ import division, print_function
import numpy as np
from pyscf.nao.m_ao_matelem import build_3dgrid
from pyscf.nao.m_dens_libnao import dens_libnao
from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
from pyscf.dft import libxc

#
#
#
def xc_scalar_ni(me, sp1,R1, sp2,R2, xc_code, deriv, **kw):
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

  grids = build_3dgrid(me, sp1, R1, sp2, R2, **kw)
  rho = dens_libnao(grids.coords, me.sv.nspin)

  THRS = 1e-12 
  if me.sv.nspin==1: 
    rho[rho<THRS] = 0.0
  elif me.sv.nspin==2:
    msk = np.logical_or(rho[:,0]<THRS, rho[:,1]<THRS)
    rho[msk,0],rho[msk,1] = 0.0,0.0

  exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho.T, spin=me.sv.nspin-1, deriv=deriv)
  ao1 = ao_eval(me.ao1, R1, sp1, grids.coords)
 
  if deriv==1 :
    xq = vxc[0] if vxc[0].ndim>1 else vxc[0].reshape((vxc[0].size,1))
  elif deriv==2:
    xq = fxc[0] if fxc[0].ndim>1 else fxc[0].reshape((fxc[0].size,1))
  else:
    print(' deriv ', deriv)
    raise RuntimeError('!deriv!')

  ao1 = np.einsum('ax,x,xq->qax', ao1, grids.weights, xq)
  ao2 = ao_eval(me.ao2, R2, sp2, grids.coords)
  return np.einsum('qax,bx->qab', ao1, ao2)
