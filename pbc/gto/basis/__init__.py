#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com> 

import os
import imp
import pyscf.gto.basis
from pyscf.pbc.gto.basis import parse_cp2k

def parse(string):
    '''Parse the basis text which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`

    Args:
        string : Blank linke and the lines of "BASIS SET" and "END" will be ignored

    Examples:

    >>> cell = gto.Cell()
    >>> cell.basis = {'C': gto.basis.parse("""
    ... C DZVP-GTH
    ...   2
    ...   2  0  1  4  2  2
    ...         4.3362376436   0.1490797872   0.0000000000  -0.0878123619   0.0000000000
    ...         1.2881838513  -0.0292640031   0.0000000000  -0.2775560300   0.0000000000
    ...         0.4037767149  -0.6882040510   0.0000000000  -0.4712295093   0.0000000000
    ...         0.1187877657  -0.3964426906   1.0000000000  -0.4058039291   1.0000000000
    ...   3  2  2  1  1
    ...         0.5500000000   1.0000000000
    ... #
    ... """)}
    '''
    return parse_cp2k.parse_str(string)

def load(basis_name, symb):
    '''Convert the basis of the given symbol to internal format

    Args:
        basis_name : str
            Case insensitive basis set name. Special characters will be removed.
        symb : str
            Atomic symbol, Special characters will be removed.

    Examples:
        Load DZVP-GTH of carbon 

    >>> cell = gto.Cell()
    >>> cell.basis = {'C': load('gth-dzvp', 'C')}
    '''
    alias = {
        'gthaugdzvp'  : 'gth-aug-dzvp.dat',
        'gthaugqzv2p' : 'gth-aug-qzv2p.dat',
        'gthaugqzv3p' : 'gth-aug-qzv3p.dat',
        'gthaugtzv2p' : 'gth-aug-tzv2p.dat',
        'gthaugtzvp'  : 'gth-aug-tzvp.dat',
        'gthdzv'      : 'gth-dzv.dat',      
        'gthdzvp'     : 'gth-dzvp.dat',     
        'gthqzv2p'    : 'gth-qzv2p.dat',    
        'gthqzv3p'    : 'gth-qzv3p.dat',    
        'gthszv'      : 'gth-szv.dat',      
        'gthtzv2p'    : 'gth-tzv2p.dat',    
        'gthtzvp'     : 'gth-tzvp.dat',     
    }

    name = basis_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    if 'gth' not in basis_name:
        return pyscf.gto.basis.load(basis_name, symb)
    basmod = alias[name]
    symb = ''.join(i for i in symb if i.isalpha())
    b = parse_cp2k.parse(os.path.join(os.path.dirname(__file__), basmod), symb)
    return b

