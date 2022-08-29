#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Analytical nuclear gradients for PBC
'''
from pyscf.pbc.grad import rhf
from pyscf.pbc.grad import rks
from pyscf.pbc.grad import uhf
from pyscf.pbc.grad import uks
from pyscf.pbc.grad import krhf
from pyscf.pbc.grad import kuhf
from pyscf.pbc.grad import krks
from pyscf.pbc.grad import kuks

from pyscf.pbc.grad.krhf import Gradients as KRHF
from pyscf.pbc.grad.kuhf import Gradients as KUHF
from pyscf.pbc.grad.krks import Gradients as KRKS
from pyscf.pbc.grad.kuks import Gradients as KUKS

grad_nuc = rhf.grad_nuc
