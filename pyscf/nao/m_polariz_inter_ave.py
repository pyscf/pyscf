from __future__ import print_function, division
import numpy as np
from pyscf import lib

def polariz_inter_ave(mf, gto, tddft, comega):
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('cmb,bn->cmn', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  vo_dip = vo_dip.reshape((3,int(vo_dip.size/3)))
  p = np.zeros((len(comega)), dtype=np.complex128)
  #print(vo_dip.shape)
  for (x,y),e in zip(tddft.xy, tddft.e):
    #print(x.shape, y.shape)
    dip = np.dot(vo_dip, np.sqrt(2.0)*(x+y)[0]) # Normalization ?
    osc_strength = (2.0/3.0)*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e))-(1.0/(w+e)))
  return p

def polariz_nonin_ave(mf, gto, comega):
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('cmb,bn->cmn', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  vo_dip = vo_dip.reshape((3,int(vo_dip.size/3)))
  p = np.zeros((len(comega)), dtype=np.complex128)
  eai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])
  for dip,e in zip(vo_dip.T,eai):
    osc_strength = (2.0/3.0)*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e[0]))-(1.0/(w+e[0])))
  return p
