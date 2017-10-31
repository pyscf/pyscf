from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf as scf_gto
from pyscf.nao import tddft_iter
from pyscf import lib

def comp_polariz_ave(mf, gto, tddft, comega):
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('cmb,bn->cmn', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  vo_dip = vo_dip.reshape((3,int(vo_dip.size/3)))
  p = np.zeros((comega.size), dtype=np.complex128)
  for (x,y),e in zip(tddft.xy, tddft.e):
    dip = np.dot(vo_dip, (x+y))
    osc_strength = (2.0/3.0)*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e))-(1.0/(w+e)))
  return p

def comp_polariz_nonin_ave(mf, gto, comega):
  gto.set_common_orig((0.0,0.0,0.0))
  ao_dip = gto.intor_symmetric('int1e_r', comp=3)
  occidx = np.where(mf.mo_occ==2)[0]
  viridx = np.where(mf.mo_occ==0)[0]
  mo_coeff = mf.mo_coeff
  mo_energy = mf.mo_energy
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('cmb,bn->cmn', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  vo_dip = vo_dip.reshape((3,int(vo_dip.size/3)))
  p = np.zeros((comega.size), dtype=np.complex128)
  eai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])
  for dip,e in zip(vo_dip.T,eai):
    osc_strength = (2.0/3.0)*(dip*dip).sum()
    for iw,w in enumerate(comega):
      p[iw] += osc_strength*((1.0/(w-e[0]))-(1.0/(w+e[0])))
  return p

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0.17    0.7    0.587''', basis = 'cc-pvdz',)

gto_mf = scf_gto.RKS(mol)
gto_mf.kernel()
gto_td = tddft.TDDFT(gto_mf)
gto_td.nstates = 90
gto_td.kernel()

nao_td  = tddft_iter(mf=gto_mf, gto=mol)

class KnowValues(unittest.TestCase):
    
  def test_tddft_gto_vs_nao_inter(self):
    """ Interacting case """
    omegas = np.linspace(0.0,2.0,150)+1j*0.04
    p_ave = -comp_polariz_ave(gto_mf, mol, gto_td, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tddft_lda.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    p_iter = -nao_td.comp_polariz_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.tddft_iter_lda.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    #print('inter', abs(p_ave-p_iter*0.5).sum()/omegas.size)
    self.assertTrue(abs(p_ave-p_iter*0.5).sum()/omegas.size< 0.02)
    
  def test_tddft_gto_vs_nao_nonin(self):
    """ Non-interacting case """
    omegas = np.linspace(0.0,2.0,150)+1j*0.04
    p_ave = -comp_polariz_nonin_ave(gto_mf, mol, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tddft_lda.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    
    nao_td  = tddft_iter(mf=gto_mf, gto=mol)
    p_iter = -nao_td.comp_nonin_polariz_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.tddft_iter_lda.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    #print('nonin', abs(p_ave-p_iter).sum()/omegas.size)
    self.assertTrue(abs(p_ave-p_iter).sum()/omegas.size< 0.03)

if __name__ == "__main__": print("Test of TDDFT GTO versus NAO"); unittest.main()
