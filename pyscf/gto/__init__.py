#!/usr/bin/env python
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

from pyscf.gto import mole
from pyscf.gto import basis
from pyscf.gto.basis import parse, load, parse_ecp, load_ecp
from pyscf.gto.mole import *
from pyscf.gto.moleintor import getints, getints_by_shell
from pyscf.gto.eval_gto import eval_gto
from pyscf.gto import ecp

parse = basis.parse
#import pyscf.gto.mole.cmd_args
