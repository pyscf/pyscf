from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf

dname = os.path.dirname(os.path.abspath(__file__))
dft = mf(label='water', cd=dname)

class KnowValues(unittest.TestCase):

  def test_dos(self):
    """ This is DoS with the KS eigenvalues """
    omegas = np.linspace(-1.0,1.0,500)+1j*0.01
    dos = dft.dos(omegas)
    data = np.array([27.2114*omegas.real, dos])
    np.savetxt('water.dos.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.dos.txt-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))
    
  def test_pdos(self):
    """ This is PDoS with SIESTA starting point """
    omegas = np.linspace(-1.0,1.0,500)+1j*0.01
    pdos = dft.pdos(omegas)
    data = np.array([27.2114*omegas.real, pdos[0], pdos[1], pdos[2]])
    np.savetxt('water.pdos.txt', data.T)
    data_ref = np.loadtxt(dname+'/water.pdos.txt-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
