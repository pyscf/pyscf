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

_windows_dll_dir_handles = []

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _loaderpath = os.path.dirname(__file__)
    _seen = set()
    _candidates = [
        _loaderpath,
        os.path.join(_loaderpath, 'deps', 'bin'),
        os.path.join(_loaderpath, 'deps', 'win64', 'bin'),
        os.path.join(_loaderpath, 'deps', 'lib'),
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(sys.prefix, 'Library', 'lib'),
    ]
    _candidates.extend(os.environ.get('PATH', '').split(os.pathsep))
    for _path in _candidates:
        if not _path:
            continue
        _path = os.path.abspath(_path)
        if _path in _seen or not os.path.isdir(_path):
            continue
        _seen.add(_path)
        try:
            _windows_dll_dir_handles.append(os.add_dll_directory(_path))
        except OSError:
            pass
    del _candidates, _loaderpath, _path, _seen

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
