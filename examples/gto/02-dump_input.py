#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Mute input dumping

Input file is dump at the beginning of the output by default. Switch it off
by setting dump_input=False. Also the command line arguments can be ignored
if parse_arg=False.

dump_input and parse_arg are the first two arguments of the Mole.build
function. They can be turned off by short notation .build(0, 0). This trick
can be found many places in the package.
'''

mol = gto.M(
    verbose = 5,
    atom = 'H  0 0 1; H 0 0 2',
    dump_input=False,
    parse_arg=False,
)

mol.build(0, 0)
