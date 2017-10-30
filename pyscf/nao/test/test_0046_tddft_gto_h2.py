from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf as scf_gto
from pyscf.nao import tddft_iter

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)

def comp_polariz_xx(mf, gto, tddft, comega):
  mol.set_common_orig((0,0,0))
  ao_dip = mol.intor_symmetric('int1e_r', comp=3)
  occ_tol = 1e-15
  occidx = np.where(mf.mo_occ>2-occ_tol)[0]
  viridx = np.where(mf.mo_occ<occ_tol)[0]
  mo_coeff = mf.mo_coeff
  orbv,orbo = mo_coeff[:,viridx], mo_coeff[:,occidx]
  vo_dip = np.einsum('cmb,bn->cmn', np.einsum('am,cab->cmb', orbv, ao_dip), orbo)
  vo_dip = vo_dip.reshape((3,int(vo_dip.size/3)))
  print(vo_dip.shape)
  pxx = np.zeros((comega.size, 3,3), dtype=np.complex128)
  for (x,y),e in zip(tddft.xy, tddft.e):
    x_dip = np.dot(vo_dip,x)
    y_dip = np.dot(vo_dip,y)
    print(x_dip.sum(),y_dip.sum(),e)
    for iw,w in enumerate(comega):
      #print(x,y,e)
      pxx[iw,:,:] = pxx[iw,:,:]
  return pxx
  

class KnowValues(unittest.TestCase):
    
  def test_tddft_gto_vs_nao(self):
    """ """
    gto_mf = scf_gto.RKS(mol)
    gto_mf.kernel()
    #print(dir(gto_mf))
    #print(gto_mf.xc)
    #print(gto_mf.pop())

    gto_td = tddft.TDDFT(gto_mf)
    gto_td.nstates = 9
    gto_td.kernel()

    omegas = np.linspace(0.0,2.0,1)+1j*0.0267
    comp_polariz_xx(gto_mf, mol, gto_td, omegas)
    
    print('Excitation energy (eV)', gto_td.e * 27.2114)
    print(dir(gto_td))
    a,b = gto_td.gen_vind(gto_mf)
    print(a)
    print(b)
    
    #print(' gto_td.xy.shape ', len(gto_td.xy))
    #print(ao_dip.shape)
    #for x,y in gto_td.xy:
    #  print(x.shape, y.shape)


    nao_td  = tddft_iter(mf=gto_mf, gto=mol)
    omegas = np.linspace(0.0,2.0,150)+1j*nao_td.eps
    pxx = -nao_td.comp_nonin(omegas).imag
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('hydrogen.tddft_iter_lda.omega.nonin.pxx.txt', data.T, fmt=['%f','%f'])
    
    pxx = -nao_td.comp_polariz_xx(omegas).imag
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('hydrogen.tddft_iter_lda.omega.inter.pxx.txt', data.T, fmt=['%f','%f'])

    #data_ref = np.loadtxt(dname+'/water.tddft_iter_lda.omega.inter.pxx.txt-ref')
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)
    #self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))

if __name__ == "__main__":
  print("Test of TDDFT GTO versus NAO")
  unittest.main()
