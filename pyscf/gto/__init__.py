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
from pyscf.gto.mole import (
    lib, param, elements, logger, cmd_args, moleintor, core_configuration, ELEMENTS,
    ELEMENTS_PROTON, charge, is_ghost_atom, CHARGE_OF, PTR_COORD, NUC_MOD_OF, PTR_ZETA,
    PTR_FRAC_CHARGE, PTR_RADIUS, ATM_SLOTS, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF,
    RADI_POWER, KAPPA_OF, SO_TYPE_OF, PTR_EXP, PTR_COEFF, BAS_SLOTS, PTR_EXPCUTOFF,
    PTR_COMMON_ORIG, PTR_RINV_ORIG, PTR_RINV_ZETA, PTR_RANGE_OMEGA, PTR_F12_ZETA,
    PTR_GTG_ZETA, NGRIDS, PTR_GRIDS, AS_RINV_ORIG_ATOM, AS_ECPBAS_OFFSET, AS_NECPBAS,
    PTR_ENV_START, NUC_POINT, NUC_GAUSS, NUC_FRAC_CHARGE, NUC_ECP, BASE, NORMALIZE_GTO,
    DISABLE_EVAL, ARGPARSE, DUMPINPUT, BSE_META, NUMBERED_ATOM_LABEL, contract,
    contracted_basis, CHECK_GEOM
)
from pyscf.gto.moleintor import getints, getints_by_shell
from pyscf.gto.eval_gto import eval_gto
from pyscf.gto import ecp

#import pyscf.gto.mole.cmd_args
