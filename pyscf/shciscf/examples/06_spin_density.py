#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: James Smith <james.smith9113@gmail.com>
import numpy as np
from pyscf import gto, scf
from pyscf.shciscf import shci

#
# Setup and run HCISCF
#

mol = gto.M(
    atom="""
    C   0.0000     0.0000    0.0000  
    H   -0.9869    0.3895    0.2153  
    H   0.8191     0.6798   -0.1969  
    H   0.1676    -1.0693   -0.0190  
""",
    spin=1,
)
ncas, nelecas = (7, 7)
mf = scf.ROHF(mol).run()
mc = shci.SHCISCF(mf, ncas, nelecas)
mc.kernel()

#
# Get our spin 1-RDMs
#
dm1 = mc.fcisolver.make_rdm1(0, ncas, nelecas)
dm_ab = mc.fcisolver.make_rdm1s()  # in MOs
dm1_check = np.linalg.norm(dm1 - np.sum(dm_ab, axis=0))
print(f"RDM1 Check = {dm1_check}\n")

#
# Use UHF tools for Spin Density Analysis
#

# Mulliken Population Analysis
dm_ab = mc.make_rdm1s()  # in AOs
scf.uhf.mulliken_spin_pop(mol, dm_ab, s=mf.get_ovlp())
print()

# Mullikan Population Analysis using Meta-Lowdin Orbitals
scf.uhf.mulliken_spin_pop_meta_lowdin_ao(mol, dm_ab, s=mf.get_ovlp())

#
# Cleanup
#
mc.fcisolver.cleanup_dice_files()
