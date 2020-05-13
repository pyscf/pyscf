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


from pyscf.nao.m_siesta_ev2ha import siesta_ev2ha

def siesta_eig(label='siesta'):
  f = open(label+'.EIG', 'r')
  f.seek(0)
  Fermi_energy_eV = float(f.readline())
  Fermi_energy_Ha = Fermi_energy_eV * siesta_ev2ha
  f.close()
  return Fermi_energy_Ha
  
  
