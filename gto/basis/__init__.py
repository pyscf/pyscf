#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

import os
import sys
if sys.version_info < (2,7):
    import imp
else:
    import importlib
from pyscf.gto.basis import parse_nwchem

ALIAS = {
    'ano'        : 'ano.dat'        ,
    'anoroosdz'  : 'roos-dz.dat'    ,
    'anoroostz'  : 'roos-tz.dat'    ,
    'roosdz'     : 'roos-dz.dat'    ,
    'roostz'     : 'roos-tz.dat'    ,
    'ccpvdz'     : 'cc-pvdz.dat'    ,
    'ccpvtz'     : 'cc-pvtz.dat'    ,
    'ccpvqz'     : 'cc-pvqz.dat'    ,
    'ccpv5z'     : 'cc-pv5z.dat'    ,
    'augccpvdz'  : 'aug-cc-pvdz.dat',
    'augccpvtz'  : 'aug-cc-pvtz.dat',
    'augccpvqz'  : 'aug-cc-pvqz.dat',
    'augccpv5z'  : 'aug-cc-pv5z.dat',
    'ccpvdzdk'   : 'cc-pvdz-dk.dat' ,
    'ccpvtzdk'   : 'cc-pvtz-dk.dat' ,
    'ccpvqzdk'   : 'cc-pvqz-dk.dat' ,
    'ccpv5zdk'   : 'cc-pv5z-dk.dat' ,
    'ccpvdzdkh'  : 'cc-pvdz-dk.dat' ,
    'ccpvtzdkh'  : 'cc-pvtz-dk.dat' ,
    'ccpvqzdkh'  : 'cc-pvqz-dk.dat' ,
    'ccpv5zdkh'  : 'cc-pv5z-dk.dat' ,
    'augccpvdzdk' : 'aug-cc-pvdz-dk.dat',
    'augccpvtzdk' : 'aug-cc-pvtz-dk.dat',
    'augccpvqzdk' : 'aug-cc-pvqz-dk.dat',
    'augccpv5zdk' : 'aug-cc-pv5z-dk.dat',
    'augccpvdzdkh': 'aug-cc-pvdz-dk.dat',
    'augccpvtzdkh': 'aug-cc-pvtz-dk.dat',
    'augccpvqzdkh': 'aug-cc-pvqz-dk.dat',
    'augccpv5zdkh': 'aug-cc-pv5z-dk.dat',
    'ccpvtzdk3'   : 'cc-pVTZ-DK3.dat'   ,
    'ccpvqzdk3'   : 'cc-pVQZ-DK3.dat'   ,
    'augccpvtzdk3': 'aug-cc-pVTZ-DK3.dat',
    'augccpvqzdk3': 'aug-cc-pVQZ-DK3.dat',
    'dyalldz'    : 'dyall_dz'       ,
    'dyallqz'    : 'dyall_qz'       ,
    'dyalltz'    : 'dyall_tz'       ,
    'faegredz'   : 'faegre_dz'      ,
    'iglo'       : 'iglo3'          ,
    'iglo3'      : 'iglo3'          ,
    '321g'       : '3-21g.dat'      ,
    '431g'       : '4-31g.dat'      ,
    '631g'       : '6-31g.dat'      ,
    '631gs'      : '6-31gs.dat'     ,
    '631gsp'     : '6-31gsp.dat'    ,
    '631gps'     : '6-31gsp.dat'    ,
    '6311g'      : '6-311g.dat'     ,
    '6311gs'     : '6-311gs.dat'    ,
    '6311gsp'    : '6-311gsp.dat'   ,
    '6311gps'    : '6-311gsp.dat'   ,
    '631g*'      : '6-31gs.dat'     ,
    '631g*+'     : '6-31gsp.dat'    ,
    '631g+*'     : '6-31gsp.dat'    ,
    '6311g*'     : '6-311gs.dat'    ,
    '6311g*+'    : '6-311gsp.dat'   ,
    '6311g+*'    : '6-311gsp.dat'   ,
    'sto3g'      : 'sto-3g.dat'     ,
    'sto6g'      : 'sto-6g.dat'     ,
    'minao'      : 'minao'          ,
    'dz'         : 'dz.dat'         ,
    'dzpdunning' : 'dzp_dunning'    ,
    'dzvp'       : 'dzvp.dat'       ,
    'dzvp2'      : 'dzvp2.dat'      ,
    'dzp'        : 'dzp.dat'        ,
    'tzp'        : 'tzp.dat'        ,
    'qzp'        : 'qzp.dat'        ,
    'adzp'       : 'adzp.dat'       ,
    'atzp'       : 'atzp.dat'       ,
    'aqzp'       : 'aqzp.dat'       ,
    'dzpdk'      : 'dzp-dkh.dat'    ,
    'tzpdk'      : 'tzp-dkh.dat'    ,
    'qzpdk'      : 'qzp-dkh.dat'    ,
    'dzpdkh'     : 'dzp-dkh.dat'    ,
    'tzpdkh'     : 'tzp-dkh.dat'    ,
    'qzpdkh'     : 'qzp-dkh.dat'    ,
    'def2svp'    : 'def2-svp.dat'   ,
    'def2svpd'   : 'def2-svpd.dat'  ,
    'def2qzvpd'  : 'def2-qzvpd.dat' ,
    'def2qzvppd' : 'def2-qzvppd.dat',
    'def2qzvpp'  : 'def2-qzvpp.dat' ,
    'def2qzvp'   : 'def2-qzvp.dat'  ,
    'def2tzvpd'  : 'def2-tzvpd.dat' ,
    'def2tzvppd' : 'def2-tzvppd.dat',
    'def2tzvpp'  : 'def2-tzvpp.dat' ,
    'def2tzvp'   : 'def2-tzvp.dat'  ,
    'tzv'        : 'tzv.dat'        ,
    'weigend'    : 'weigend_cfit.dat',
    'weigend+etb': 'weigend_cfit.dat',
    'demon'      : 'demon_cfit.dat' ,
    'ahlrichs'   : 'ahlrichs_cfit.dat',
    'ccpvtzfit'  : 'cc-pvtz_fit.dat',
    'ccpvdzfit'  : 'cc-pvdz_fit.dat',
    'ccpwcvtzmp2fit': 'cc-pwCVTZ_MP2FIT.dat',
    'ccpvqzmp2fit': 'cc-pVQZ_MP2FIT.dat',
    'ccpv5zmp2fit': 'cc-pV5Z_MP2FIT.dat',
    'augccpwcvtzmp2fit': 'aug-cc-pwCVTZ_MP2FIT.dat',
    'augccpvqzmp2fit': 'aug-cc-pVQZ_MP2FIT.dat',
    'augccpv5zmp2fit': 'aug-cc-pV5Z_MP2FIT.dat',
    'ccpcvdz'    : ('cc-pvdz.dat', 'cc-pCVDZ.dat'),
    'ccpcvtz'    : ('cc-pvtz.dat', 'cc-pCVTZ.dat'),
    'ccpcvqz'    : ('cc-pvqz.dat', 'cc-pCVQZ.dat'),
    #'ccpcv5z'    : 'cc-pCV5Z.dat',
    'ccpcv6z'    : 'cc-pCV6Z.dat',
    'ccpwcvdz'   : ('cc-pvdz.dat', 'cc-pwCVDZ.dat'),
    'ccpwcvtz'   : 'cc-pwCVTZ.dat',
    'ccpwcvqz'   : 'cc-pwCVQZ.dat',
    'ccpwcv5z'   : 'cc-pwCV5Z.dat',
    'ccpwcvdzdk' : ('cc-pvdz.dat', 'cc-pwCVDZ-DK.dat'),
    'ccpwcvtzdk' : 'cc-pwCVTZ-DK.dat',
    'ccpwcvqzdk' : 'cc-pwCVQZ-DK.dat',
    'ccpwcvtzdk3': 'cc-pwCVTZ-DK3.dat',
    'ccpwcvqzdk3': 'cc-pwCVQZ-DK3.dat',
    'augccpwcvtzdk' : 'aug-cc-pwCVTZ-DK.dat',
    'augccpwcvqzdk' : 'aug-cc-pwCVQZ-DK.dat',
    'augccpwcvtzdk3': 'aug-cc-pwCVTZ-DK3.dat',
    'augccpwcvqzdk3': 'aug-cc-pwCVQZ-DK3.dat',
    'dgaussa1cfit': 'DgaussA1_dft_cfit.dat',
    'dgaussa1xfit': 'DgaussA1_dft_xfit.dat',
    'dgaussa2cfit': 'DgaussA2_dft_cfit.dat',
    'dgaussa2xfit': 'DgaussA2_dft_xfit.dat',
    'ccpvdzpp'   : 'cc-pvdz-pp.dat' ,
    'ccpvtzpp'   : 'cc-pvtz-pp.dat' ,
    'ccpvqzpp'   : 'cc-pvqz-pp.dat' ,
    'ccpv5zpp'   : 'cc-pv5z-pp.dat' ,
    'crenbl'     : 'crenbl.dat'     ,
    'crenbs'     : 'crenbs.dat'     ,
    'lanl2dz'    : 'lanl2dz.dat'    ,
    'lanl2tz'    : 'lanl2tz.dat'    ,
    'lanl08'     : 'lanl08.dat'     ,
    'sbkjc'      : 'sbkjc.dat'      ,
    'stuttgart'  : 'stuttgart_dz.dat',
    'stuttgartdz': 'stuttgart_dz.dat',
    'stuttgartrlc': 'stuttgart_dz.dat',
    'stuttgartrsc': 'stuttgart_rsc.dat',
    'ccpwcvdzpp' : 'cc-pwCVDZ-PP.dat',
    'ccpwcvtzpp' : 'cc-pwCVTZ-PP.dat',
    'ccpwcvqzpp' : 'cc-pwCVQZ-PP.dat',
    'ccpwcv5zpp' : 'cc-pwCV5Z-PP.dat',
    'ccpvdzppnr' : 'cc-pVDZ-PP-NR.dat',
    'ccpvtzppnr' : 'cc-pVTZ-PP-NR.dat',
    'augccpvdzpp': ('cc-pvdz-pp.dat', 'aug-cc-pVDZ-PP.dat'),
    'augccpvtzpp': ('cc-pvtz-pp.dat', 'aug-cc-pVTZ-PP.dat'),
    'augccpvqzpp': ('cc-pvqz-pp.dat', 'aug-cc-pVQZ-PP.dat'),
    'augccpv5zpp': ('cc-pv5z-pp.dat', 'aug-cc-pV5Z-PP.dat'),
# Burkatzki-Filippi-Dolg pseudo potential
    'bfdvdz'     : 'bfd_vdz.dat',
    'bfdvtz'     : 'bfd_vtz.dat',
    'bfdvqz'     : 'bfd_vqz.dat',
    'bfdv5z'     : 'bfd_v5z.dat',
    'bfd'        : 'bfd_pp.dat',
    'bfdpp'      : 'bfd_pp.dat',
#
    'ccpcvdzf12optri': 'f12-basis/cc-pCVDZ-F12-OptRI.dat',
    'ccpcvtzf12optri': 'f12-basis/cc-pCVTZ-F12-OptRI.dat',
    'ccpcvqzf12optri': 'f12-basis/cc-pCVQZ-F12-OptRI.dat',
    'ccpvdzf12optri' : 'f12-basis/cc-pVDZ-F12-OptRI.dat',
    'ccpvtzf12optri' : 'f12-basis/cc-pVTZ-F12-OptRI.dat',
    'ccpvqzf12optri' : 'f12-basis/cc-pVQZ-F12-OptRI.dat',
    'ccpv5zf12'      : 'f12-basis/cc-pV5Z-F12.dat',
    'ccpvdzf12rev2'  : 'f12-basis/cc-pVDZ-F12rev2.dat',
    'ccpvtzf12rev2'  : 'f12-basis/cc-pVTZ-F12rev2.dat',
    'ccpvqzf12rev2'  : 'f12-basis/cc-pVQZ-F12rev2.dat',
    'ccpv5zf12rev2'  : 'f12-basis/cc-pV5Z-F12rev2.dat',
    'ccpvdzf12nz'    : 'f12-basis/cc-pVDZ-F12-nZ.dat',
    'ccpvtzf12nz'    : 'f12-basis/cc-pVTZ-F12-nZ.dat',
    'ccpvqzf12nz'    : 'f12-basis/cc-pVQZ-F12-nZ.dat',
    'augccpvdzoptri' : 'f12-basis/aug-cc-pVDZ-OptRI.dat',
    'augccpvtzoptri' : 'f12-basis/aug-cc-pVTZ-OptRI.dat',
    'augccpvqzoptri' : 'f12-basis/aug-cc-pVQZ-OptRI.dat',
    'augccpv5zoptri' : 'f12-basis/aug-cc-pV5Z-OptRI.dat',
}

def parse(string):
    '''Parse the NWChem format basis or ECP text, return an internal basis (ECP)
    format which can be assigned to :attr:`Mole.basis` or :attr:`Mole.ecp`

    Args:
        string : Blank linke and the lines of "BASIS SET" and "END" will be ignored

    Examples:

    >>> mol = gto.Mole()
    >>> mol.basis = {'O': gto.basis.parse("""
    ... #BASIS SET: (6s,3p) -> [2s,1p]
    ... C    S
    ...      71.6168370              0.15432897
    ...      13.0450960              0.53532814
    ...       3.5305122              0.44463454
    ... C    SP
    ...       2.9412494             -0.09996723             0.15591627
    ...       0.6834831              0.39951283             0.60768372
    ...       0.2222899              0.70011547             0.39195739
    ... """)}
    '''
    if 'ECP' in string:
        return parse_nwchem.parse_ecp(string)
    else:
        return parse_nwchem.parse(string)

def parse_ecp(string):
    return parse_nwchem.parse_ecp(string)

def load(filename_or_basisname, symb):
    '''Convert the basis of the given symbol to internal format

    Args:
        filename_or_basisname : str
            Case insensitive basis set name. Special characters will be removed.
            or a string of "path/to/file" which stores the basis functions
        symb : str
            Atomic symbol, Special characters will be removed.

    Examples:
        Load STO 3G basis of carbon to oxygen atom

    >>> mol = gto.Mole()
    >>> mol.basis = {'O': load('sto-3g', 'C')}
    '''

    if os.path.isfile(filename_or_basisname):
        # read basis from given file
        try:
            return parse_nwchem.load(filename_or_basisname, symb)
        except RuntimeError:
            with open(filename_or_basisname, 'r') as fin:
                return parse_nwchem.parse(fin.read())

    name = filename_or_basisname.lower().replace(' ', '').replace('-', '').replace('_', '')
    if name not in ALIAS:
        try:
            return parse(filename_or_basisname)
        except IndexError:
            raise RuntimeError('basis %s not found' % filename_or_basisname)
    basmod = ALIAS[name]
    symb = ''.join([i for i in symb if i.isalpha()])
    if 'dat' in basmod:
        b = parse_nwchem.load(os.path.join(os.path.dirname(__file__), basmod), symb)
    elif isinstance(basmod, tuple):
        b = []
        for f in basmod:
            b += parse_nwchem.load(os.path.join(os.path.dirname(__file__), f), symb)
    else:
        if sys.version_info < (2,7):
            fp, pathname, description = imp.find_module(basmod, __path__)
            mod = imp.load_module(name, fp, pathname, description)
            b = mod.__getattribute__(symb)
            fp.close()
        else:
            mod = importlib.import_module('.'+basmod, __package__)
            b = mod.__getattribute__(symb)
    return b

def load_ecp(filename_or_basisname, symb):
    '''Convert the basis of the given symbol to internal format
    '''

    if os.path.isfile(filename_or_basisname):
        # read basis from given file
        try:
            return parse_nwchem.load_ecp(filename_or_basisname, symb)
        except RuntimeError:
            with open(filename_or_basisname, 'r') as fin:
                return parse_nwchem.parse_ecp(fin.read())

    name = filename_or_basisname.lower().replace(' ', '').replace('-', '').replace('_', '')
    if name not in ALIAS:
        return parse_ecp(filename_or_basisname)
    basmod = ALIAS[name]
    symb = ''.join([i for i in symb if i.isalpha()])
    return parse_nwchem.load_ecp(os.path.join(os.path.dirname(__file__), basmod), symb)

