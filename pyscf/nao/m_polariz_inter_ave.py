from __future__ import print_function, division
import numpy as np

def polariz_inter_xx(mf, gto, tddft, comega):
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)[0]
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('am,ab,bn->mn', orbv, ao_dip, orbo).reshape([len(viridx)*len(occidx)])

  p = np.zeros((len(comega)), dtype=np.complex128)
  #print(vo_dip.shape)
  for (x,y),e in zip(tddft.xy, tddft.e):
    #print(x.shape, y.shape)
    dip = np.dot(vo_dip, np.sqrt(2.0)*(x+y)[0]) # Normalization ?
    osc_strength = 2.0*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e))-(1.0/(w+e)))
  return p


def polariz_freq_osc_strength(t2e, t2osc, comega):

  p = np.zeros((len(comega)), dtype=np.complex128)
  for osc,e in zip(t2osc, t2e):
    for iw,w in enumerate(comega):
      p[iw] += osc*((1.0/(w-e))-(1.0/(w+e)))
  return p
    

def polariz_inter_ave(mf, gto, tddft, comega):
  assert mf.mo_occ.ndim==1
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  no,nv = orbo.shape[1],orbv.shape[1]
  xov2dip = np.einsum('cmb,bn->cnm', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  p = np.zeros((len(comega)), dtype=np.complex128)
  for (x,y),e in zip(tddft.xy, tddft.e):
    dip = np.sqrt(2.0)*np.einsum('xov,ov->x', xov2dip, x+y)
    osc_strength = (2.0/3.0)*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e))-(1.0/(w+e)))
  return p

def polariz_nonin_ave(mf, gto, comega):
  from pyscf import lib

  ns,n = mf.mo_occ.ndim, mf.mol.nao_nr() # number of spin channels, number of orbitals

  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)

  mo_occ = mf.mo_occ.reshape((ns,n))
  mo_coeff = np.asarray(mf.mo_coeff).reshape((ns,n,n))
  mo_energy = np.asarray(mf.mo_energy).reshape((ns,n))

  p = np.zeros((len(comega)), dtype=np.complex128) # Result

  for s,occ in enumerate(mo_occ):
    occidx,viridx = np.where(occ>0.1)[0],np.where(occ==0)[0] # Weak condition!!!
    ma2c,nb2c = mo_coeff[s,:,viridx], mo_coeff[s,:,occidx]
    vo_dip = np.einsum('xmb,nb->xmn', np.einsum('ma,xab->xmb', ma2c, ao_dip), nb2c)
    vo_dip = vo_dip.reshape((3,vo_dip.size//3))
    eai = lib.direct_sum('a-i->ai', mo_energy[s,viridx], mo_energy[s,occidx])
    eai = eai.reshape(eai.size)
    for dip,e in zip(vo_dip.T,eai):
      osc_strength = (2.0/3.0/ns)*(dip*dip).sum() 
      for iw,w in enumerate(comega):
        p[iw] += osc_strength*((1.0/(w-e))-(1.0/(w+e))) # Occupations differences are missing!
      
  return p
