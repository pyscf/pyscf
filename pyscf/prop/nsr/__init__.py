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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.prop.nsr import rhf
from pyscf.prop.nsr import uhf
RHF = rhf.NSR
UHF = uhf.NSR

try:
    from pyscf.prop.nsr import rks
    from pyscf.prop.nsr import uks
    RKS = rks.NSR
    UKS = uks.NSR
except ImportError:
    pass
