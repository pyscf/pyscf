from __future__ import print_function, division
import unittest
from pyscf import gto
import numpy as np

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_prdred(self):
    """  """
    from pyscf.nao import system_vars_c
    from pyscf.nao.m_prod_talman import prod_talman_c
    sv = system_vars_c().init_pyscf_gto(mol)
    self.assertEqual(sv.sp2charge[0], 1)
    pt = prod_talman_c(sv.ao_log)
    jtb,clbdtb,lbdtb=pt.prdred_terms(2,4)
    self.assertEqual(len(jtb), 295)
    self.assertEqual(pt.lbdmx, 8)


  def test_prdred_eval(self):
    from pyscf.nao.m_prod_talman import prod_talman_c
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    from pyscf.nao.m_csphar_talman_libnao import csphar_talman_libnao as csphar_jt
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao import ao_log_c
    from numpy import sqrt, zeros, array

    import os
    
    dname = os.path.dirname(os.path.abspath(__file__))
    sp2ion = []
    sp2ion.append(siesta_ion_xml(dname+'/H.ion.xml'))
    sp2ion.append(siesta_ion_xml(dname+'/O.ion.xml'))

    aos = ao_log_c().init_ao_log_ion(sp2ion, nr=512, rmin=0.0025)
    pt = prod_talman_c(aos)

    spa,mua,spb,mub =0,0,1,1
    la,lb = aos.sp_mu2j[spa][mua],aos.sp_mu2j[spb][mub]
    oas,oaf = aos.sp_mu2s[spa][mua],aos.sp_mu2s[spa][mua+1]
    obs,obf = aos.sp_mu2s[spb][mub],aos.sp_mu2s[spb][mub+1]
    
    jtb,clbdtb,lbdtb=pt.prdred_terms(la,lb)
    ra,rb,rcen = array([0.0,0.0,-0.5]),array([0.0,0.0,0.5]), zeros(3)

    jtb,clbdtb,lbdtb,rhotb = \
      pt.prdred_libnao(aos.psi_log[spa][mua,:],la,ra,aos.psi_log[spb][mub,:],lb,rb, rcen)

    coords = np.array([[0.110, 0.260, 0.22]])
    rscal  = sqrt(sum((coords[0] - rcen)**2))
    aosa,aosb = ao_eval(aos, ra, spa, coords), ao_eval(aos, rb, spb, coords)

    ylm = csphar_jt(coords[0,:], clbdtb.max())
     
    #print('\n test_prdred_eval')
    
    for oa in range(oas,oaf):
      ma = oas-oa-la
      for ob in range(obs,obf):
        mb = obs-ob-lb
        ffr = pt.prdred_further(la,ma,lb,mb,rcen,jtb,clbdtb,lbdtb,rhotb)

        ffr_vals = np.zeros(ffr.shape[0], dtype=np.complex128)
        for iclbd,ff in enumerate(ffr): ffr_vals[iclbd] = pt.log_interp(ff, rscal)
    
        prdval = 0.0+0.0j
        for clbd,ffr_val in enumerate(ffr_vals): prdval = prdval + ffr_val*ylm[clbd*(clbd+1)]

        #print(ma,mb,oa,ob,ffr_vals)
        #print(prdval, aosb[oa,0]*aosa[ob,0])

if __name__ == "__main__":
  unittest.main()
