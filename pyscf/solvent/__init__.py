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

from pyscf.solvent import ddcosmo
from pyscf.solvent.ddcosmo import DDCOSMO
from pyscf.solvent.ddpcm import DDPCM

for_scf = ddcosmo.ddcosmo_for_scf

def ddCOSMO(method):
    return ddcosmo.ddcosmo_for_scf(method)

def ddPCM(method):
    return ddpcm.ddpcm_for_scf(method)
