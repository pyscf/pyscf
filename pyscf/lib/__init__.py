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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
C extensions and helper functions
'''

from pyscf.lib import parameters
param = parameters
# Bind the submodule here rather than relying on pyscf.gto.mole importing it.
# Code such as pyscf.fci.direct_spin1_symm reaches for lib.exceptions, which
# only resolved because some other module happened to have imported it first.
from pyscf.lib import exceptions
from pyscf.lib import numpy_helper
from pyscf.lib import linalg_helper
from pyscf.lib import scipy_helper
from pyscf.lib import logger
from pyscf.lib import misc
from pyscf.lib.misc import *
from pyscf.lib.numpy_helper import *
from pyscf.lib.linalg_helper import *
from pyscf.lib.scipy_helper import *
from pyscf.lib import chkfile
from pyscf.lib import diis
from pyscf.lib.misc import StreamObject
