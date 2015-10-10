#!/usr/bin/env python

import sys
from functools import reduce
import numpy
import scipy.linalg
from pyscf.scf import chkfile
from pyscf.scf import hf

def mulliken(filename, key='scf'):
    '''Reading scf/mcscf information from chkfile, then do Mulliken population
    analysis for the density matrix
    '''
    if key.lower() == 'mcscf':
        mol = chkfile.load_mol(filename)
        mo_coeff = chkfile.load(filename, 'mcscf/mo_coeff')
        mo_occ = chkfile.load(filename, 'mcscf/mo_occ')
    else:
        mol, mf = chkfile.load_scf(filename)
        mo_coeff = mf['mo_coeff']
        mo_occ = mf['mo_occ']
    dm = numpy.dot(mo_coeff*mo_occ, mo_coeff.T)
    hf.mulliken_meta(mol, dm)

def dump_mo(filename, key='scf'):
    '''Read scf/mcscf information from chkfile, then dump the orbital
    coefficients.
    '''
    from pyscf.tools import dump_mat
    if key.lower() == 'mcscf':
        mol = chkfile.load_mol(filename)
        mo_coeff = chkfile.load(filename, 'mcscf/mo_coeff')
    else:
        mol, mf = chkfile.load_scf(filename)
        mo_coeff = mf['mo_coeff']
    dump_mat.dump_mo(mol, mo_coeff)

def molden(filename, key='scf'):
    '''Read scf/mcscf information from chkfile, then convert the scf/mcscf
    orbitals to molden format.
    '''
    from pyscf.tools import molden
    molden.from_chkfile(filename+'.molden', filename, key+'/mo_coeff')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('%s chkfile_name [pop|mo|molden] [scf|mcscf]' % sys.argv[0])
        exit()
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
