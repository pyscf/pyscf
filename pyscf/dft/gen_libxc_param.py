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

import os
import sys
ldpath = os.path.abspath(os.path.join(__file__, '..', '..', 'lib', 'deps', 'lib'))
if ldpath not in os.environ['LD_LIBRARY_PATH']:
    sys.stderr.write(f'Set\n\tLD_LIBRARY_PATH={ldpath}:$LD_LIBRARY_PATH\n'
                     'and rerun the script\n')
    exit()

pypath = os.path.join(__file__, '..', '..', 'lib', 'build', 'deps', 'src', 'libxc')
sys.path.insert(0, os.path.abspath(pypath))
import pylibxc

for xcname in pylibxc.util.xc_available_functional_names():
    f = pylibxc.LibXCFunctional(xcname, 1)
    f_id = f.get_number()
    ref = f.get_references()
    key = f"'{xcname.upper()}'"
    print(f"{key:<31s}: {f_id:<3d}, # {ref[0]}")
    for r in ref[1:]:
        print(f"                                     # {r}")
