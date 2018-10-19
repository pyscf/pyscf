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
from pyscf.nao.ao_log import ao_log

try:
  from ase import Atoms
  from gpaw import GPAW
  fname = os.path.dirname(os.path.abspath(__file__))+'/h2o.gpw'

  if os.path.isfile(fname):
    calc = GPAW(fname, txt=None) # read previous calculation if the file exists
  else:
    from gpaw import PoissonSolver
    atoms = Atoms('H2O', positions=[[0.0,-0.757,0.587], [0.0,+0.757,0.587], [0.0,0.0,0.0]])
    atoms.center(vacuum=3.5)
    convergence = {'density': 1e-7}     # Increase accuracy of density for ground state
    poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)     # Increase accuracy of Poisson Solver and apply multipole corrections up to l=1
    calc = GPAW(basis='dzp', xc='LDA', h=0.3, nbands=23, convergence=convergence, poissonsolver=poissonsolver, mode='lcao', txt=None)     # nbands must be equal to norbs (in this case 23)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()    # Do SCF the ground state
    calc.write(fname, mode='all') # write DFT output

except:
  calc = None

class KnowValues(unittest.TestCase):

  def test_ao_log_after_gpaw(self):
    """ init ao_log with it radial orbitals from GPAW """
    if calc is None: return
    self.assertTrue(hasattr(calc, 'setups'))
    aos = ao_log(setups=calc.setups)
    self.assertEqual(aos.nr, 1024)

    #print(aos.nr)
    #print(aos.rr[0])
    #print(aos.rr[-1])

if __name__ == "__main__": unittest.main()
