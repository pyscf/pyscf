from pyscf.nao.m_rsphar import rsphar
from pyscf.nao.m_rsphar_libnao import rsphar as rsphar_libnao
import numpy as np
import sys

from timeit import default_timer as timer

rvecs = np.array([[ 0.0, 0.0, 0.0],
                  [ 0.1, 0.1, 0.1],
                  [-0.1, 0.1, 0.1],
                  [ 0.1,-0.1, 0.1],
                  [-0.1,-0.1, 0.1],
                  [ 0.1, 0.1,-0.1],
                  [-0.1, 0.1,-0.1],
                  [ 0.1,-0.1,-0.1],
                  [-0.1,-0.1,-0.1],
                  ])

rvecs = np.random.rand(50000,3)
rvecs = np.add(rvecs, -0.5)

lmax = 4
rsh1 = np.zeros(((lmax+1)**2))
rsh2 = np.zeros(((lmax+1)**2))

start = timer()
for i,rvec in enumerate(rvecs):
  rsphar(rvec, lmax, rsh1)
#  rsphar_libnao(rvec, lmax, rsh2)
#  if not np.allclose(rsh1, rsh2): print(i, rsh1, rsh2)

end = timer()
print(end-start)

