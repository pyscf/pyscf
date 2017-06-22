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
    sp2ion.append(siesta_ion_xml(dname+'/O.ion.xml'))

    aos = ao_log_c().init_ao_log_ion(sp2ion)
    jmx = aos.jmx
    pt = prod_talman_c(aos)

    spa,mua,spb,mub =0,0,0,0
    la,lb = aos.sp_mu2j[spa][mua],aos.sp_mu2j[spb][mub]
    phia,phib = aos.psi_log[spa][mua,:],aos.psi_log[spb][mub,:]
    rav,rbv,rcv = array([0.0,0.0,-1.0]),array([0.0,0.0,1.0]), zeros(3)

    jtb,clbdtb,lbdtb=pt.prdred_terms(la,lb)
    jtb,clbdtb,lbdtb,rhotb = pt.prdred_libnao(phia,la,rav,phib,lb,rbv,rcv)

    coord = np.array([0.0, 0.0, 0.22]) # Point at which we will compute the expansion and the original product
    ylma,ylmb,ylmc = csphar_jt(coord-rav, jmx), csphar_jt(coord-rbv, jmx),csphar_jt(coord+rcv, pt.lbdmx)
    ras,rbs,rcs = sqrt(sum((coord-rav)**2)),sqrt(sum((coord-rbv)**2)),sqrt(sum((coord+rcv)**2))

    print('\n test_prdred_eval')
    for ma in range(-la,la+1):
      aovala = ylma[la*(la+1)+ma]*pt.log_interp(phia, ras)
      for mb in range(-lb,lb+1):
        aovalb = ylmb[lb*(lb+1)+mb]*pt.log_interp(phib, rbs)

        ffr,m = pt.prdred_further(la,ma,lb,mb,rcv,jtb,clbdtb,lbdtb,rhotb)
        ffr_vals = zeros(pt.lbdmx+1, dtype=np.complex128)
        for iclbd,ff in enumerate(ffr): ffr_vals[iclbd] = pt.log_interp(ff, rcs)
        prdval = 0.0
        for j,ffr_val in enumerate(ffr_vals): prdval = prdval + ffr_val*ylmc[j*(j+1)+m]
        
        print(ma,mb, aovala, aovalb, aovala*aovalb, prdval)
    

if __name__ == "__main__":
  unittest.main()
