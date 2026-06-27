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

import os
import sys

# Keep the bundled support DLL search path narrow on Windows so the wheel can
# load libcint/libxc from deps/bin without duplicating those large DLLs in pyscf/lib.
_windows_dll_dir_handles = []
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _loaderpath = os.path.dirname(__file__)
    _deps_bin = os.path.join(_loaderpath, 'deps', 'bin')
    if os.path.isdir(_deps_bin):
        try:
            _windows_dll_dir_handles.append(os.add_dll_directory(_deps_bin))
        except OSError:
            pass
    del _deps_bin, _loaderpath

from pyscf.lib import parameters
param = parameters
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
