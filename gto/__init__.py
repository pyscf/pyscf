#!/usr/bin/env python
# -*- coding: utf-8
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
