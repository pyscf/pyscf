# TODO: refactor the code before adding to FEATURES list by PySCF-1.5 release
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
# 1. code style
#   * Remove the unused modules: numpy, scipy, gto, dft, ...
#

# We should get the lib import working for now let's just do a quick TDSCF.
#from pyscf.tdscf import bo
import numpy as np
import scipy
import scipy.linalg
from pyscf import gto, dft, scf, ao2mo
from tdscf import *
