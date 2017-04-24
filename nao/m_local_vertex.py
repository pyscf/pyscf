from __future__ import print_function
from __future__ import division
import numpy as np
import sys
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from scipy.linalg import eigh

#
#
#
def ij2pack(i,j):
  ma = max(i,j)
  return ma*(ma+1)//2+min(i,j)

#
#
#
class local_vertex_c(ao_matelem_c):
  '''
    Constructor of the local product functions and the product vertex coefficients.
  '''
  def __init__(self, ao_log):
    ao_matelem_c.__init__(self, ao_log)
    nr = self.nr
    kk = self.kk
    self.dkappa_pp = 4*np.pi*np.log( kk[nr-1]/kk[0])/(nr-1)*kk
    self.c2r_c = c2r_c(2*self.jmx) # local vertex c2r[:,:] coefficients
    
  #
  #
  #
  def get_local_vertex(self, sp):
    """
      Constructor of vertex for a given specie
      Args:
        sp : specie number
      Result:
        Dictionary with the product functions, vertex coefficients and eigenvalues
        in each angular-momentum "sektor"
          dominant products functions: j2xff
          dominant vertex coefficients (angular part of): j2xww
          eigenvalues of Coulomb metric : j2eva
      
    """
    assert(sp>-1)

    jmx_sp = np.amax(self.sp_mu2j[sp])
    j2nf=np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in self.sp2info[sp]:
      for mu2,j2,s2,f2 in self.sp2info[sp]:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2): 
          j2nf[j] = j2nf[j] + 1
    
    j_p2mus = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j_p2js  = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j2p = np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in self.sp2info[sp]:
      for mu2,j2,s2,f2 in self.sp2info[sp]:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2):
          j_p2mus[j][j2p[j]] = [mu1,mu2]
          j_p2js[j][j2p[j]] = [j1,j2]
          j2p[j]+=1

    no = self.sp2norbs[sp]
    nmu = len(self.sp2mults[sp])
    pack2ff = np.zeros((nmu*(nmu+1)//2,self.nr), dtype='float64') # storage for original products
    for mu2 in self.sp2mults[sp]:
      for mu1 in range(mu2+1):
        pack2ff[ij2pack(mu1,mu2),:] = self.psi_log[sp][mu1,:]*self.psi_log[sp][mu2,:]
    
    j2xff = [] # Storage for dominant product's functions (list of numpy arrays: x*f(r)*f(r))
    j2xww = [] # Storage for dominant product's vertex (angular part of: x*wigner*wigner)
    j2eva = [] # Storage for eigenvalues in each angular momentum "sektor"
    hc_c2r = np.conj(self.c2r_c._c2r)
    c2r_jm = self.c2r_c._j
    for j,dim in enumerate(j2nf): # Metrik ist dim * dim in diesem Sektor 

      metric = np.zeros((dim,dim), dtype='float64')
      for level_1 in range(dim):
        ff12_p = self.sbt(pack2ff[ ij2pack( *j_p2mus[j][level_1] ),:], j, 1)
        for level_2 in range(level_1+1):
          ff34_p = self.sbt(pack2ff[ ij2pack( *j_p2mus[j][level_2] ),:], j, 1)
          metric[level_2,level_1]=metric[level_1,level_2]=sum(ff12_p*ff34_p*self.dkappa_pp)  # Coulomb Metrik enthaelt Faktor 1/p**2

      eva,x=eigh(metric)
      j2eva.append(eva)

      if j==0 and sp==0:
        x[:,0] = [0.379391313,  -0.109740384,   0.425570488,  -0.523525417,   0.256656945,  -0.398414046,  -0.405245155 ]
        x[:,1] = [0.387586743,   9.81706604E-02,0.452922672, -4.16178033E-02,-0.530645430, -6.12854362E-02, 0.589853883 ]
        x[:,2] = [0.414766550,   0.228208750,   0.355080962,   0.639254749,   0.325311482,   0.298978776,  -0.214346588 ]
        x[:,3] = [0.368397653,  -0.136905298,  -0.296508014,  -0.247163445,   0.591030359,   0.222216651,   0.545743704 ] 
        x[:,4] = [0.379803926,   0.160379171,  -0.286661327,  -0.349554688,  -0.379521132,   0.592939496,  -0.360625565 ]
        x[:,5] = [0.421171069,   0.334853470,  -0.549279153,   0.223333612,  -0.101834126,  -0.588435948,   -4.77030762E-02]
        x[:,6] = [0.276504159,  -0.877296388,  -0.125972241,   0.287030131,  -0.207344413,   -3.01289931E-02, -0.108359329 ]
        x = x.transpose()
      elif j==1 and sp==0:
        x[:,0] = [0.42352697891624486,       0.17730215080624570 ,      -8.4975873977810479E-002,  0.71102789529995780,       0.43918804206781481,       0.28900024584975176     ]
        x[:,1] = [0.44944958866067281,      -0.21076173752565167 ,       2.3797677501368575E-002,  0.33016689279828010,      -0.43413710477535150,      -0.67492445869939055     ]
        x[:,2] = [0.43428535051812267,      -0.20237976224894444 ,      0.11978226254428881     , -0.11718650923491836,      -0.55684550935891952,       0.65748100846893132     ]
        x[:,3] = [0.48167637480533726,      -0.45567740058051126 ,     -0.33050134060457997     , -0.47076474894115550,       0.47190990699223184,       -8.2440470947864455E-002]
        x[:,4] = [0.32227384383985724,       0.78603879057007375 ,     -0.39337982104642277     , -0.30006451543909013,      -0.16023204655243106,       -8.8441412930972130E-002]
        x[:,5] = [0.30682835468589398,       0.24018117217224128 ,      0.84491784288123473     , -0.24502717152848849,       0.24533374739102393,      -0.11855902192352212]
        x = x.transpose()

      xff = np.zeros((dim,self.nr))   #!!!! Jetzt dominante Orbitale bilden
      for domi in range(dim):  
        for n in range(dim):
          xff[domi,:] = xff[domi,:] + x[n,domi]*pack2ff[ij2pack(*j_p2mus[j][n]),:]
      j2xff.append(xff)
      
      #xe = np.zeros((dim,dim))
      #for i,ev in enumerate(eva): xe[:,i] = x[:,i]*ev
      #metric1 = np.matmul(x,xe.transpose())
      #print(sum(sum(abs(metric1-metric1.transpose()))), sum(sum(abs(metric1-metric))))

      kinematical_vertex = np.zeros((dim, 2*j+1, no, no), dtype='float64')
      for num,[[mu1,mu2], [j1,j2]] in enumerate(zip(j_p2mus[j],j_p2js[j])):
        if j<abs(j1-j2) or j>j1+j2 : continue
        for m1,o1 in zip(range(-j1,j1+1), range(self.sp_mu2s[sp][mu1],self.sp_mu2s[sp][mu1]+2*j1+1)):
          for m2,o2 in zip(range(-j2,j2+1), range(self.sp_mu2s[sp][mu2],self.sp_mu2s[sp][mu2]+2*j2+1)):
            m=m1+m2
            if abs(m)>j: continue
            i3y=self.get_gaunt(j1,m1,j2,m2)*(-1.0)**m
            kinematical_vertex[num,j+m,o2,o1] = kinematical_vertex[num,j+m,o1,o2] = i3y[j-abs(j1-j2)]
      
      xww = np.zeros((dim, 2*j+1, no, no), dtype='float64')
      for domi in range(dim):
        xww0 = np.zeros((2*j+1, no, no), dtype='float64')
        for num in range(dim): 
          xww0[:,:,:] = xww0[:,:,:] + x[num,domi]*kinematical_vertex[num,:,:,:]
                
        xww1 = np.zeros((2*j+1, no, no), dtype='complex128')
        for m in range(-j,j+1):
          for m1 in range(-j,j+1):
            xww1[j+m,:,:]=xww1[j+m,:,:]+hc_c2r[c2r_jm+m1,c2r_jm+m]*xww0[j+m1,:,:]

        for m in range(-j,j+1):
          xww2 = np.zeros((no,no), dtype='complex128')
          for mu1,j1,s1,f1 in self.sp2info[sp]:
            for mu2,j2,s2,f2 in self.sp2info[sp]:
              for m1,o1 in zip(range(-j1,j1+1), range(self.sp_mu2s[sp][mu1],self.sp_mu2s[sp][mu1]+2*j1+1)):
                for m2,o2 in zip(range(-j2,j2+1), range(self.sp_mu2s[sp][mu2],self.sp_mu2s[sp][mu2]+2*j2+1)):
                  for n1,p1 in zip(range(-j1,j1+1), range(self.sp_mu2s[sp][mu1],self.sp_mu2s[sp][mu1]+2*j1+1)):
                    for n2,p2 in zip(range(-j2,j2+1), range(self.sp_mu2s[sp][mu2],self.sp_mu2s[sp][mu2]+2*j2+1)):
                      xww2[o1,o2]=xww2[o1,o2]+self._c2r[m1,n1]*self._c2r[m2,n2] * xww1[j+m,p1,p2]
          xww[domi,j+m,:,:] = xww2[:,:].real
        print(j, domi, (sum(sum(sum(xww[domi,:,:,:])))))
            
      if j==1 : raise SystemError('j=1')
      j2xww.append(xww)

    return {"j2xww": j2xww, "j2xff": j2xff, "j2eva": j2eva }
