#!/usr/bin/env python

'''
Serialization for a PySCF object.
'''

# Most methods can be pickled
import pickle
import pyscf

mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
mf = mol.RKS(xc='pbe')
s = pickle.dumps(mf)
mf1 = pickle.loads(s)

# Dynamically generated classes cannot be serialized by the standard pickle module.
# In this case, the third party packages cloudpickle or dill support the
# dynamical classes.
import cloudpickle
mf = mol.RHF().density_fit().x2c().newton()
s = cloudpickle.dumps(mf)
mf1 = cloudpickle.loads(s)
