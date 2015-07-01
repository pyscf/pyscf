#!/usr/bin/env python

import sys
from functools import reduce
import numpy
import scipy.linalg
from pyscf.scf import chkfile
from pyscf.scf import hf

def mulliken(filename, key='scf'):
    mol, mf = chkfile.load_scf(filename)
    if key.lower() == 'mcscf':
        mo_coeff = chkfile.load(filename, 'mcscf/mo_coeff')
        mo_occ = chkfile.load(filename, 'mcscf/mo_occ')
    else:
        mo_coeff = mf['mo_coeff']
        mo_occ = mf['mo_occ']
    dm = numpy.dot(mo_coeff*mo_occ, mo_coeff.T)
    hf.mulliken_meta(mol, dm)

def dump_mo(filename, key='scf'):
    from pyscf.tools import dump_mat
    mol, mf = chkfile.load_scf(filename)
    if key.lower() == 'mcscf':
        mo_coeff = chkfile.load(filename, 'mcscf/mo_coeff')
    else:
        mo_coeff = mf['mo_coeff']
    dump_mat.dump_mo(mol, mo_coeff)

def molden(filename, key='scf'):
    from pyscf.tools import molden
    molden.from_chkfile(filename+'.molden', filename, key+'/mo_coeff')

if __name__ == '__main__':
    filename = sys.argv[1]
    fndic = {'pop': mulliken,
             'mo': dump_mo,
             'molden': molden}
    fn = fndic[sys.argv[2].lower()]
    if len(sys.argv) > 3:
        key = sys.argv[3]
        fn(filename, key)
    else:
        fn(filename)
