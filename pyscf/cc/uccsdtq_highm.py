# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

# Author: Yuhao Chen <chenyuhao@shu.edu.cn>

'''Full iterative UCCSDTQ with full native spin-block T3/T4 storage.

This class selects the full-storage paths of the shared UCCSDTQ solver.
The parent uses upstream uccsdt_highm for R1--R3 and passes the selected
layout to the quadruples engines, vectorization, and denominator routines.
Sharing those equations avoids a second copy of the quadruples term table;
this is a storage variant, not an independent equation implementation.
'''

from pyscf.cc import uccsdtq


class UCCSDTQ(uccsdtq.UCCSDTQ):
    do_tri_max_t = property(lambda self: False)
