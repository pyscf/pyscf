from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix


#
#
#
class fireball_hsx():

  def __init__(self, nao, **kw):
    from pyscf.nao.m_fireball_get_HS_dat import fireball_get_HS_dat
    from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
    
    self.cd = nao.cd
    atom2s = nao.atom2s
    
    self.fname = fname = kw['fname'] if 'fname' in kw else 'HS.dat'   
    i2aoao,i2h,i2s,i2x = fireball_get_HS_dat(self.cd, fname)
    ni = len(i2aoao)
    i2a = np.zeros((ni), dtype=int)
    i2b = np.zeros((ni), dtype=int)
    for i,aoao in enumerate(i2aoao):
      i2a[i] = atom2s[aoao[0]-1]+aoao[1]-1
      i2b[i] = atom2s[aoao[2]-1]+aoao[3]-1

    m = atom2s[-1]
    n = atom2s[-1]
    self.spin2h4_csr = [csr_matrix((i2h, (i2a, i2b)), shape=(m, n))]
    self.s4_csr = csr_matrix((i2s, (i2a, i2b)), shape=(m, n))
    self.x4_csr = [
       csr_matrix((i2x[:,0], (i2a, i2b)), shape=(m, n)),
       csr_matrix((i2x[:,1], (i2a, i2b)), shape=(m, n)),
       csr_matrix((i2x[:,2], (i2a, i2b)), shape=(m, n)) ]

    self.orb_sc2orb_uc = np.array(list(range(atom2s[-1])), dtype=int)

    o2m = nao.get_orb2m()
    _siesta2blanko_csr(o2m, self.s4_csr, self.orb_sc2orb_uc)

    _siesta2blanko_csr(o2m, self.spin2h4_csr[0], self.orb_sc2orb_uc)



  def deallocate(self):
    del self.spin2h4_csr
    del self.s4_csr
