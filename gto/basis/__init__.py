#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

def importbas(basis_name, symb):
    # dict to map the basis name and basis file
    alias = {
        'ano'        : 'ano'        ,
        'ccpv5z'     : 'cc_pv5z'    ,
        'ccpvdz'     : 'cc_pvdz'    ,
        'augccpvdz'  : 'aug_cc_pvdz',
        'ccpvqz'     : 'cc_pvqz'    ,
        'augccpvqz'  : 'aug_cc_pvqz',
        'ccpvtz'     : 'cc_pvtz'    ,
        'augccpvtz'  : 'aug_cc_pvtz',
        'dyalldz'    : 'dyall_dz'   ,
        'dyallqz'    : 'dyall_qz'   ,
        'dyalltz'    : 'dyall_tz'   ,
        'faegredz'   : 'faegre_dz'  ,
        'iglo'       : 'iglo3'      ,
        'iglo3'      : 'iglo3'      ,
        '321g'       : 'p3_21g'     ,
        '431g'       : 'p4_31g'     ,
        '631g'       : 'p6_31g'     ,
        '631gs'      : 'p6_31gs'    ,
        '6311g'      : 'p6_311g'    ,
        '6311gs'     : 'p6_311gs'   ,
        '6311gsp'    : 'p6_311gsp'  ,
        '6311gps'    : 'p6_311gsp'  ,
        '321g'       : 'p3_21g'     ,
        '431g'       : 'p4_31g'     ,
        '631g'       : 'p6_31g'     ,
        '631g*'      : 'p6_31gs'    ,
        '6311g'      : 'p6_311g'    ,
        '6311g*'     : 'p6_311gs'   ,
        '6311g*+'    : 'p6_311gsp'  ,
        '6311g+*'    : 'p6_311gsp'  ,
        'sto3g'      : 'sto_3g'     ,
        'sto6g'      : 'sto_6g'     ,
        'minao'      : 'minao'      ,
        'dzpdunning' : 'dzp_dunning',
        'dzp'        : 'dzp'        ,
        'tzp'        : 'tzp'        ,
        'qzp'        : 'qzp'        ,
        'def2qzvpd'  : 'def2_qzvpd' ,
        'def2qzvppd' : 'def2_qzvppd',
        'def2qzvpp'  : 'def2_qzvpp' ,
        'def2qzvp'   : 'def2_qzvp'  ,
        'def2tzvpd'  : 'def2_tzvpd' ,
        'def2tzvppd' : 'def2_tzvppd',
        'def2tzvpp'  : 'def2_tzvpp' ,
        'def2tzvp'   : 'def2_tzvp'  ,
    }
    basmod = alias[basis_name]
    #mod = __import__(basmod, globals())
    mod = __import__(basmod, globals={'__path__': __path__, '__name__': __name__})
    v = mod.__getattribute__(symb)
    return mod.__getattribute__(symb)

