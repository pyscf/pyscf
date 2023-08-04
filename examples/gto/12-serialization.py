#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pickle
from pyscf import gto

'''
Serialize Mole object

Mole.pack function transform the Mole object to a Python dict.  It can be
serialized using the standard serialization module in python, eg pickle.

There is a hacky way to serialize the Mole object for broadcasting or saving.
Use format or str function to convert the Mole-python-dict to a string.
Later, use the built-in eval function and Mole.unpack function to restore the
Mole object.
'''

mol = gto.M(
    atom = '''
O        0.000000    0.000000    0.117790
H1       0.000000    0.755453   -0.471161
H2       0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
)

# In Python pickled format
ar = pickle.dumps(format(mol.pack()))
mol1 = gto.unpack(eval(pickle.loads(ar)))

# In JSON format
ar = mol.dumps()
mol1 = gto.loads(ar)

