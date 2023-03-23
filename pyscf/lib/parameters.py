# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
PySCF environment variables are defined in this module.


Scratch directory
-----------------

The PySCF scratch directory is specified by :data:`TMPDIR`.  Its default value
is the same to the system-wide environment variable ``TMPDIR``.  It can be
overwritten by the environment variable ``PYSCF_TMPDIR``.  Another place to set
``TMPDIR`` is the global configuration file (see :ref:`global_config`).


.. _max_mem:

Maximum memory
--------------

The variable :data:`MAX_MEMORY` defines the maximum memory that PySCF can be
used in the calculation.  Its unit is MB.  The default value is 4000 MB.  It can
be overwritten by the system-wide environment variable ``PYSCF_MAX_MEMORY``.
``MAX_MEMORY`` can also be set in :ref:`global_config` file.

.. note:: Some calculations may exceed the max_memory limit, especially
  when the attribute :attr:`Mole.incore_anyway` was set.
'''


from pyscf.data.nist import LIGHT_SPEED, BOHR
from pyscf.data.elements import ELEMENTS, ELEMENTS_PROTON, NUC
from pyscf import __config__

MAX_MEMORY = getattr(__config__, 'MAX_MEMORY', 4000)  # MB
TMPDIR = getattr(__config__, 'TMPDIR', '.')
OUTPUT_DIGITS = getattr(__config__, 'OUTPUT_DIGITS', 5)
OUTPUT_COLS   = getattr(__config__, 'OUTPUT_COLS', 5)
LOOSE_ZERO_TOL = getattr(__config__, 'LOOSE_ZERO_TOL', 1e-9)
LARGE_DENOM = getattr(__config__, 'LARGE_DENOM', 1e14)

L_MAX = 8
ANGULAR = 'spdfghik'
ANGULARMAP = {'s': 0,
              'p': 1,
              'd': 2,
              'f': 3,
              'g': 4,
              'h': 5,
              'i': 6,
              'k': 7}

REAL_SPHERIC = (
    ('',),
    ('x', 'y', 'z'),
    ('xy', 'yz', 'z^2', 'xz', 'x2-y2',),
    ('-3', '-2', '-1', '+0', '+1', '+2', '+3'),
    ('-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4'),
    ('-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5'),
    ('-6', '-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5', '+6'),
)

VERBOSE_DEBUG  = 5
VERBOSE_INFO   = 4
VERBOSE_NOTICE = 3
VERBOSE_WARN   = 2
VERBOSE_ERR    = 1
VERBOSE_QUIET  = 0
VERBOSE_CRIT   = -1
VERBOSE_ALERT  = -2
VERBOSE_PANIC  = -3
