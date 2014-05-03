#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

from ano import *
from cc_pv5z import *
from cc_pvdz import *
from cc_pvqz import *
from cc_pvtz import *
from dyall_dz import *
from dyall_qz import *
from dyall_tz import *
from faegre_dz import *
from iglo import *
from p3_21g import *
from p4_31g import *
from p6_311g import *
from p6_31g import *
from sto_3g import *
from sto_6g import *
from minao import *


# dict to map the basis name and basis file
alias = {
    'ano'       : ano,
    'ccpv5z'    : cc_pv5z,
    'ccpvdz'    : cc_pvdz,
    'augccpvdz' : aug_cc_pvdz,
    'ccpvqz'    : cc_pvqz,
    'augccpvqz' : aug_cc_pvqz,
    'ccpvtz'    : cc_pvtz,
    'augccpvtz' : aug_cc_pvtz,
    'dyalldz'   : dyall_dz,
    'dyallqz'   : dyall_qz,
    'dyalltz'   : dyall_tz,
    'faegredz'  : faegre_dz,
    'iglo3'     : iglo3,
    'p321g'     : p3_21g,
    'p431g'     : p4_31g,
    'p631g'     : p6_31g,
    'p631gs'    : p6_31gs,
    'p6311g'    : p6_311g,
    'p6311gs'   : p6_311gs,
    'p6311gsp'  : p6_311gsp,
    'sto3g'     : sto_3g,
    'sto6g'     : sto_6g,
    '321g'      : p3_21g,
    '431g'      : p4_31g,
    '631g'      : p6_31g,
    '6311g'     : p6_311g,
    '631g*'     : p6_31gs,
    '6311g*'    : p6_311gs,
    '6311g*+'   : p6_311gsp,
    '631g**'    : p6_31gs,
    '6311g**'   : p6_311gs,
    '6311g**++' : p6_311gsp,
    'minao'     : minao,
}
