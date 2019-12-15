# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf

try:
  from ase import Atoms
  from gpaw import GPAW
  fname = os.path.dirname(os.path.abspath(__file__))+'/h2o.gpw'

  from gpaw import PoissonSolver
  atoms = Atoms('H2O', positions=[[0.0,-0.757,0.587], [0.0,+0.757,0.587], [0.0,0.0,0.0]])
  atoms.center(vacuum=3.5)
  convergence = {'density': 1e-7}     # Increase accuracy of density for ground state
  poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)     # Increase accuracy of Poisson Solver and apply multipole corrections up to l=1

  # hgh and sg15 setups works only with minimal basis set!
  calc = GPAW(xc='LDA', h=0.3, nbands=6,
        convergence=convergence, poissonsolver=poissonsolver,
        mode='lcao', txt=None, setups="hgh")     # nbands must be equal to norbs (in this case 6)
  atoms.set_calculator(calc)
  atoms.get_potential_energy()    # Do SCF the ground state
  calc.write(fname, mode='all') # write DFT output

except:
  calc = None



class KnowValues(unittest.TestCase):

  def test_sv_after_gpaw(self):
    """ init ao_log with radial orbitals from GPAW """
    if calc is None: return

    self.assertTrue(hasattr(calc, 'setups'))
    sv = mf(gpaw=calc, gen_pb=False)
    self.assertEqual(sv.ao_log.nr, 1024)
    over = sv.overlap_coo().toarray()
    error = sv.hsx.check_overlaps(over)
    self.assertLess(error, 1e-4)
    #print("overlap error: ", error/over.size)

    #print("Pyscf    gpaw")
    #for py, gp in zip(over[:, 0], sv.wfsx.overlaps[:, 0]):
    #    print("{0:.5f}    {1:.5f}".format(py, gp))
    #np.savetxt("Pyscf.overlaps", over)
    #np.savetxt("gpaw.overlaps", sv.wfsx.overlaps,)


if __name__ == "__main__": unittest.main()
