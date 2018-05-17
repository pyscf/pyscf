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

from pyscf.pbc.dft.gen_grid import UniformGrids, BeckeGrids
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import uks
from pyscf.pbc.dft import roks
from pyscf.pbc.dft import krks
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft import kroks

RKS = rks.RKS
UKS = uks.UKS
ROKS = roks.ROKS

KRKS = krks.KRKS
KUKS = kuks.KUKS
KROKS = kroks.KROKS

