#!/usr/bin/env python
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

import os
import sys
from os.path import join
if sys.version_info < (2,7):
    import imp
else:
    import importlib
from pyscf.gto.basis import parse_nwchem
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf import __config__

ALIAS = {
    'ano'        : 'ano.dat'        ,
    'anorcc'     : 'ano.dat'        ,
    'anoroosdz'  : 'roos-dz.dat'    ,
    'anoroostz'  : 'roos-tz.dat'    ,
    'roosdz'     : 'roos-dz.dat'    ,
    'roostz'     : 'roos-tz.dat'    ,
    'ccpvdz'     : 'cc-pvdz.dat'    ,
    'ccpvtz'     : 'cc-pvtz.dat'    ,
    'ccpvqz'     : 'cc-pvqz.dat'    ,
    'ccpv5z'     : 'cc-pv5z.dat'    ,
    'ccpvdpdz'   : 'cc-pvdpdz.dat'  ,
    'augccpvdz'  : 'aug-cc-pvdz.dat',
    'augccpvtz'  : 'aug-cc-pvtz.dat',
    'augccpvqz'  : 'aug-cc-pvqz.dat',
    'augccpv5z'  : 'aug-cc-pv5z.dat',
    'augccpvdpdz': 'aug-cc-pvdpdz.dat',
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
    'ccpvdzjkfit' : 'cc-pvdz-jkfit.dat' ,
    'ccpvtzjkfit' : 'cc-pvtz-jkfit.dat' ,
    'ccpvqzjkfit' : 'cc-pvqz-jkfit.dat' ,
    'ccpv5zjkfit' : 'cc-pv5z-jkfit.dat' ,
    'ccpvdzri'    : 'cc-pvdz-ri.dat'    ,
    'ccpvtzri'    : 'cc-pvtz-ri.dat'    ,
    'ccpvqzri'    : 'cc-pvqz-ri.dat'    ,
    'ccpv5zri'    : 'cc-pv5z-ri.dat'    ,
    'augccpvdzjkfit' : 'aug-cc-pvdz-jkfit.dat' ,
    'augccpvdzpjkfit': 'aug-cc-pvdzp-jkfit.dat',
    'augccpvtzjkfit' : 'aug-cc-pvtz-jkfit.dat' ,
    'augccpvqzjkfit' : 'aug-cc-pvqz-jkfit.dat' ,
    'augccpv5zjkfit' : 'aug-cc-pv5z-jkfit.dat' ,
    'heavyaugccpvdzjkfit' : 'heavy-aug-cc-pvdz-jkfit.dat',
    'heavyaugccpvtzjkfit' : 'heavy-aug-cc-pvtz-jkfit.dat',
    'heavyaugccpvdzri' : 'heavy-aug-cc-pvdz-ri.dat',
    'heavyaugccpvtzri' : 'heavy-aug-cc-pvtz-ri.dat',
    'augccpvdzri'    : 'aug-cc-pvdz-ri.dat'    ,
    'augccpvdzpri'   : 'aug-cc-pvdzp-ri.dat'   ,
    'augccpvqzri'    : 'aug-cc-pvqz-ri.dat'    ,
    'augccpvtzri'    : 'aug-cc-pvtz-ri.dat'    ,
    'augccpv5zri'    : 'aug-cc-pv5z-ri.dat'    ,
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
    '321++g'     : join('pople-basis', '3-21++G.dat'   ),
    '321++g*'    : join('pople-basis', '3-21++Gs.dat'  ),
    '321++gs'    : join('pople-basis', '3-21++Gs.dat'  ),
    '321g'       : join('pople-basis', '3-21G.dat'     ),
    '321g*'      : join('pople-basis', '3-21Gs.dat'    ),
    '321gs'      : join('pople-basis', '3-21Gs.dat'    ),
    '431g'       : join('pople-basis', '4-31G.dat'     ),
    '631++g'     : join('pople-basis', '6-31++G.dat'   ),
    '631++g*'    : join('pople-basis', '6-31++Gs.dat'  ),
    '631++gs'    : join('pople-basis', '6-31++Gs.dat'  ),
    '631++g**'   : join('pople-basis', '6-31++Gss.dat' ),
    '631++gss'   : join('pople-basis', '6-31++Gss.dat' ),
    '631+g'      : join('pople-basis', '6-31+G.dat'    ),
    '631+g*'     : join('pople-basis', '6-31+Gs.dat'   ),
    '631+gs'     : join('pople-basis', '6-31+Gs.dat'   ),
    '631+g**'    : join('pople-basis', '6-31+Gss.dat'  ),
    '631+gss'    : join('pople-basis', '6-31+Gss.dat'  ),
    '6311++g'    : join('pople-basis', '6-311++G.dat'  ),
    '6311++g*'   : join('pople-basis', '6-311++Gs.dat' ),
    '6311++gs'   : join('pople-basis', '6-311++Gs.dat' ),
    '6311++g**'  : join('pople-basis', '6-311++Gss.dat'),
    '6311++gss'  : join('pople-basis', '6-311++Gss.dat'),
    '6311+g'     : join('pople-basis', '6-311+G.dat'   ),
    '6311+g*'    : join('pople-basis', '6-311+Gs.dat'  ),
    '6311+gs'    : join('pople-basis', '6-311+Gs.dat'  ),
    '6311+g**'   : join('pople-basis', '6-311+Gss.dat' ),
    '6311+gss'   : join('pople-basis', '6-311+Gss.dat' ),
    '6311g'      : join('pople-basis', '6-311G.dat'    ),
    '6311g*'     : join('pople-basis', '6-311Gs.dat'   ),
    '6311gs'     : join('pople-basis', '6-311Gs.dat'   ),
    '6311g**'    : join('pople-basis', '6-311Gss.dat'  ),
    '6311gss'    : join('pople-basis', '6-311Gss.dat'  ),
    '631g'       : join('pople-basis', '6-31G.dat'     ),
    '631g*'      : join('pople-basis', '6-31Gs.dat'    ),
    '631gs'      : join('pople-basis', '6-31Gs.dat'    ),
    '631g**'     : join('pople-basis', '6-31Gss.dat'   ),
    '631gss'     : join('pople-basis', '6-31Gss.dat'   ),
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
    'def2tzvpd'  : 'def2-tzvpd.dat' ,
    'def2tzvppd' : 'def2-tzvppd.dat',
    'def2tzvpp'  : 'def2-tzvpp.dat' ,
    'def2tzvp'   : 'def2-tzvp.dat'  ,
    'def2qzvpd'  : 'def2-qzvpd.dat' ,
    'def2qzvppd' : 'def2-qzvppd.dat',
    'def2qzvpp'  : 'def2-qzvpp.dat' ,
    'def2qzvp'   : 'def2-qzvp.dat'  ,
    'def2svpjfit'    : 'def2-universal-jfit.dat',
    'def2svpjkfit'   : 'def2-universal-jkfit.dat',
    'def2tzvpjfit'   : 'def2-universal-jfit.dat',
    'def2tzvpjkfit'  : 'def2-universal-jkfit.dat',
    'def2tzvppjfit'  : 'def2-universal-jfit.dat',
    'def2tzvppjkfit' : 'def2-universal-jkfit.dat',
    'def2qzvpjfit'   : 'def2-universal-jfit.dat',
    'def2qzvpjkfit'  : 'def2-universal-jkfit.dat',
    'def2qzvppjfit'  : 'def2-universal-jfit.dat',
    'def2qzvppjkfit' : 'def2-universal-jkfit.dat',
    'def2universaljfit'  : 'def2-universal-jfit.dat',
    'def2universaljkfit' : 'def2-universal-jkfit.dat',
    'def2svpri'      : 'def2-svp-ri.dat'     ,
    'def2svpdri'     : 'def2-svpd-ri.dat'    ,
    'def2tzvpri'     : 'def2-tzvp-ri.dat'    ,
    'def2tzvpdri'    : 'def2-tzvpd-ri.dat'   ,
    'def2tzvppri'    : 'def2-tzvpp-ri.dat'   ,
    'def2tzvppdri'   : 'def2-tzvppd-ri.dat'  ,
    'def2qzvpri'     : 'def2-qzvp-ri.dat'    ,
    'def2qzvppri'    : 'def2-qzvpp-ri.dat'   ,
    'def2qzvppdri'   : 'def2-qzvppd-ri.dat'  ,
    'tzv'        : 'tzv.dat'        ,
    'weigend'     : 'def2-universal-jfit.dat',
    'weigend+etb' : 'def2-universal-jfit.dat',
    'weigendcfit' : 'def2-universal-jfit.dat',
    'weigendjfit' : 'def2-universal-jfit.dat',
    'weigendjkfit': 'def2-universal-jkfit.dat',
    'demon'      : 'demon_cfit.dat' ,
    'demoncfit'  : 'demon_cfit.dat' ,
    'ahlrichs'   : 'ahlrichs_cfit.dat',
    'ahlrichscfit': 'ahlrichs_cfit.dat',
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
    'ccpcv5z'    : 'cc-pCV5Z.dat',
    'ccpcv6z'    : 'cc-pCV6Z.dat',
    'ccpwcvdz'   : 'cc-pwCVDZ.dat',
    'ccpwcvtz'   : 'cc-pwCVTZ.dat',
    'ccpwcvqz'   : 'cc-pwCVQZ.dat',
    'ccpwcv5z'   : 'cc-pwCV5Z.dat',
    'ccpwcvdzdk' : 'cc-pwCVDZ-DK.dat',
    'ccpwcvtzdk' : 'cc-pwCVTZ-DK.dat',
    'ccpwcvqzdk' : 'cc-pwCVQZ-DK.dat',
    'ccpwcv5zdk' : 'cc-pwCV5Z-DK.dat',
    'ccpwcvtzdk3': 'cc-pwCVTZ-DK3.dat',
    'ccpwcvqzdk3': 'cc-pwCVQZ-DK3.dat',
    'augccpwcvdz': 'aug-cc-pwcvdz.dat',
    'augccpwcvtz': 'aug-cc-pwcvtz.dat',
    'augccpwcvqz': 'aug-cc-pwcvqz.dat',
    'augccpwcv5z': 'aug-cc-pwcv5z.dat',
    'augccpwcvtzdk' : 'aug-cc-pwCVTZ-DK.dat',
    'augccpwcvqzdk' : 'aug-cc-pwCVQZ-DK.dat',
    'augccpwcv5zdk' : 'aug-cc-pwcv5z-dk.dat',
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
    # Stuttgart ECP http://www.tc.uni-koeln.de/PP/clickpse.en.html
    'stuttgart'  : 'stuttgart_dz.dat',
    'stuttgartdz': 'stuttgart_dz.dat',
    'stuttgartrlc': 'stuttgart_dz.dat',
    'stuttgartrsc': 'stuttgart_rsc.dat',
    'stuttgartrsc_mdf': 'cc-pvdz-pp.dat',
    #'stuttgartrsc_mwb': 'stuttgart_rsc.dat',
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
    'pc0' : 'pc-0.dat',
    'pc1' : 'pc-1.dat',
    'pc2' : 'pc-2.dat',
    'pc3' : 'pc-3.dat',
    'pc4' : 'pc-4.dat',
    'augpc0' : 'aug-pc-0.dat',
    'augpc1' : 'aug-pc-1.dat',
    'augpc2' : 'aug-pc-2.dat',
    'augpc3' : 'aug-pc-3.dat',
    'augpc4' : 'aug-pc-4.dat',
    'pcseg0' : 'pcseg-0.dat',
    'pcseg1' : 'pcseg-1.dat',
    'pcseg2' : 'pcseg-2.dat',
    'pcseg3' : 'pcseg-3.dat',
    'pcseg4' : 'pcseg-4.dat',
    'augpcseg0' : 'aug-pcseg-0.dat',
    'augpcseg1' : 'aug-pcseg-1.dat',
    'augpcseg2' : 'aug-pcseg-2.dat',
    'augpcseg3' : 'aug-pcseg-3.dat',
    'augpcseg4' : 'aug-pcseg-4.dat',
    'sarcdkh'   : 'sarc-dkh2.dat',
# Burkatzki-Filippi-Dolg pseudo potential
    'bfdvdz'     : 'bfd_vdz.dat',
    'bfdvtz'     : 'bfd_vtz.dat',
    'bfdvqz'     : 'bfd_vqz.dat',
    'bfdv5z'     : 'bfd_v5z.dat',
    'bfd'        : 'bfd_pp.dat',
    'bfdpp'      : 'bfd_pp.dat',
#
    'ccpcvdzf12optri': os.path.join('f12-basis', 'cc-pCVDZ-F12-OptRI.dat'),
    'ccpcvtzf12optri': os.path.join('f12-basis', 'cc-pCVTZ-F12-OptRI.dat'),
    'ccpcvqzf12optri': os.path.join('f12-basis', 'cc-pCVQZ-F12-OptRI.dat'),
    'ccpvdzf12optri' : os.path.join('f12-basis', 'cc-pVDZ-F12-OptRI.dat' ),
    'ccpvtzf12optri' : os.path.join('f12-basis', 'cc-pVTZ-F12-OptRI.dat' ),
    'ccpvqzf12optri' : os.path.join('f12-basis', 'cc-pVQZ-F12-OptRI.dat' ),
    'ccpv5zf12'      : os.path.join('f12-basis', 'cc-pV5Z-F12.dat'       ),
    'ccpvdzf12rev2'  : os.path.join('f12-basis', 'cc-pVDZ-F12rev2.dat'   ),
    'ccpvtzf12rev2'  : os.path.join('f12-basis', 'cc-pVTZ-F12rev2.dat'   ),
    'ccpvqzf12rev2'  : os.path.join('f12-basis', 'cc-pVQZ-F12rev2.dat'   ),
    'ccpv5zf12rev2'  : os.path.join('f12-basis', 'cc-pV5Z-F12rev2.dat'   ),
    'ccpvdzf12nz'    : os.path.join('f12-basis', 'cc-pVDZ-F12-nZ.dat'    ),
    'ccpvtzf12nz'    : os.path.join('f12-basis', 'cc-pVTZ-F12-nZ.dat'    ),
    'ccpvqzf12nz'    : os.path.join('f12-basis', 'cc-pVQZ-F12-nZ.dat'    ),
    'augccpvdzoptri' : os.path.join('f12-basis', 'aug-cc-pVDZ-OptRI.dat' ),
    'augccpvtzoptri' : os.path.join('f12-basis', 'aug-cc-pVTZ-OptRI.dat' ),
    'augccpvqzoptri' : os.path.join('f12-basis', 'aug-cc-pVQZ-OptRI.dat' ),
    'augccpv5zoptri' : os.path.join('f12-basis', 'aug-cc-pV5Z-OptRI.dat' ),
# All-electron basis designed for periodic calculations, available in Crystal
    'pobtzvp'       :  'pob-tzvp.dat',
    'pobtzvpp'      :  'pob-tzvpp.dat',
    'crystalccpvdz' :  'crystal-cc-pvdz.dat',
# ccECP 
    'ccecp'         : join('ccecp-basis', 'ccECP', 'ccECP.dat'   ),
    'ccecpccpvdz'   : join('ccecp-basis', 'ccECP', 'ccECP_cc-pVDZ.dat'),
    'ccecpccpvtz'   : join('ccecp-basis', 'ccECP', 'ccECP_cc-pVTZ.dat'),
    'ccecpccpvqz'   : join('ccecp-basis', 'ccECP', 'ccECP_cc-pVQZ.dat'),
    'ccecpccpv5z'   : join('ccecp-basis', 'ccECP', 'ccECP_cc-pV5Z.dat'),
    'ccecpccpv6z'   : join('ccecp-basis', 'ccECP', 'ccECP_cc-pV6Z.dat'),
    'ccecpaugccpvdz': join('ccecp-basis', 'ccECP', 'ccECP_aug-cc-pVDZ.dat'),
    'ccecpaugccpvtz': join('ccecp-basis', 'ccECP', 'ccECP_aug-cc-pVTZ.dat'),
    'ccecpaugccpvqz': join('ccecp-basis', 'ccECP', 'ccECP_aug-cc-pVQZ.dat'),
    'ccecpaugccpv5z': join('ccecp-basis', 'ccECP', 'ccECP_aug-cc-pV5Z.dat'),
    'ccecpaugccpv6z': join('ccecp-basis', 'ccECP', 'ccECP_aug-cc-pV6Z.dat'),
# ccECP_He_core 
    'ccecphe'         : join('ccecp-basis', 'ccECP_He_core', 'ccECP.dat'   ),
    'ccecpheccpvdz'   : join('ccecp-basis', 'ccECP_He_core', 'ccECP_cc-pVDZ.dat'),
    'ccecpheccpvtz'   : join('ccecp-basis', 'ccECP_He_core', 'ccECP_cc-pVTZ.dat'),
    'ccecpheccpvqz'   : join('ccecp-basis', 'ccECP_He_core', 'ccECP_cc-pVQZ.dat'),
    'ccecpheccpv5z'   : join('ccecp-basis', 'ccECP_He_core', 'ccECP_cc-pV5Z.dat'),
    'ccecpheccpv6z'   : join('ccecp-basis', 'ccECP_He_core', 'ccECP_cc-pV6Z.dat'),
    'ccecpheaugccpvdz': join('ccecp-basis', 'ccECP_He_core', 'ccECP_aug-cc-pVDZ.dat'),
    'ccecpheaugccpvtz': join('ccecp-basis', 'ccECP_He_core', 'ccECP_aug-cc-pVTZ.dat'),
    'ccecpheaugccpvqz': join('ccecp-basis', 'ccECP_He_core', 'ccECP_aug-cc-pVQZ.dat'),
    'ccecpheaugccpv5z': join('ccecp-basis', 'ccECP_He_core', 'ccECP_aug-cc-pV5Z.dat'),
    'ccecpheaugccpv6z': join('ccecp-basis', 'ccECP_He_core', 'ccECP_aug-cc-pV6Z.dat'),
# ccECP_reg
    'ccecpreg'         : join('ccecp-basis', 'ccECP_reg', 'ccECP.dat'   ),
    'ccecpregccpvdz'   : join('ccecp-basis', 'ccECP_reg', 'ccECP_cc-pVDZ.dat'),
    'ccecpregccpvtz'   : join('ccecp-basis', 'ccECP_reg', 'ccECP_cc-pVTZ.dat'),
    'ccecpregccpvqz'   : join('ccecp-basis', 'ccECP_reg', 'ccECP_cc-pVQZ.dat'),
    'ccecpregccpv5z'   : join('ccecp-basis', 'ccECP_reg', 'ccECP_cc-pV5Z.dat'),
    'ccecpregaugccpvdz': join('ccecp-basis', 'ccECP_reg', 'ccECP_aug-cc-pVDZ.dat'),
    'ccecpregaugccpvtz': join('ccecp-basis', 'ccECP_reg', 'ccECP_aug-cc-pVTZ.dat'),
    'ccecpregaugccpvqz': join('ccecp-basis', 'ccECP_reg', 'ccECP_aug-cc-pVQZ.dat'),
    'ccecpregaugccpv5z': join('ccecp-basis', 'ccECP_reg', 'ccECP_aug-cc-pV5Z.dat'),
#spin-orbit ECPs
    'ecpds10mdfso' : os.path.join('soecp', 'ECPDS10MDFSO.dat'),
    'ecpds28mdfso' : os.path.join('soecp', 'ECPDS28MDFSO.dat'),
    'ecpds28mwbso' : os.path.join('soecp', 'ECPDS28MWBSO.dat'),
    'ecpds46mdfso' : os.path.join('soecp', 'ECPDS46MDFSO.dat'),
    'ecpds60mdfso' : os.path.join('soecp', 'ECPDS60MDFSO.dat'),
    'ecpds60mwbso' : os.path.join('soecp', 'ECPDS60MWBSO.dat'),
    'ecpds78mdfso' : os.path.join('soecp', 'ECPDS78MDFSO.dat'),
    'ecpds92mdfbso' : os.path.join('soecp', 'ECPDS92MDFBSO.dat'),
    'ecpds92mdfbqso' : os.path.join('soecp', 'ECPDS92MDFBQSO.dat'),
# dyall's sets
    'dyall2zp' : 'dyall-basis.dyall_2zp',
    'dyall3zp' : 'dyall-basis.dyall_3zp',
    'dyall4zp' : 'dyall-basis.dyall_4zp',
    'dyallaae2z' : 'dyall-basis.dyall_aae2z',
    'dyallaae3z' : 'dyall-basis.dyall_aae3z',
    'dyallaae4z' : 'dyall-basis.dyall_aae4z',
    'dyallacv2z' : 'dyall-basis.dyall_acv2z',
    'dyallacv3z' : 'dyall-basis.dyall_acv3z',
    'dyallacv4z' : 'dyall-basis.dyall_acv4z',
    'dyallae2z' : 'dyall-basis.dyall_ae2z',
    'dyallae3z' : 'dyall-basis.dyall_ae3z',
    'dyallae4z' : 'dyall-basis.dyall_ae4z',
    'dyallav2z' : 'dyall-basis.dyall_av2z',
    'dyallav3z' : 'dyall-basis.dyall_av3z',
    'dyallav4z' : 'dyall-basis.dyall_av4z',
    'dyallcv2z' : 'dyall-basis.dyall_cv2z',
    'dyallcv3z' : 'dyall-basis.dyall_cv3z',
    'dyallcv4z' : 'dyall-basis.dyall_cv4z',
    'dyallv2z' : 'dyall-basis.dyall_v2z',
    'dyallv3z' : 'dyall-basis.dyall_v3z',
    'dyallv4z' : 'dyall-basis.dyall_v4z',
}

def _is_pople_basis(basis):
    return (basis.startswith('631') or
            basis.startswith('321') or
            basis.startswith('431'))

_BASIS_DIR = os.path.dirname(__file__)

def _parse_pople_basis(basis, symb):
    if '(' in basis:
        mbas = basis[:basis.find('(')]
        extension = basis[basis.find('(')+1:basis.find(')')]
    else:
        mbas = basis
        extension = ''

    # polarized functions are defined based on pople basis name prefix like
    # 6-31G, 6-311G etc.
    basename = mbas[0] + '-' + mbas[1:].upper()
    basename = basename.replace('+', '').replace('*', '')
    pathtmp = join('pople-basis', basename + '-polarization-%s.dat')
    def convert(s):
        if len(s) == 0:
            return []
        elif s[0].isalpha():
            return [pathtmp % s[0]] + convert(s[1:])
        else:
            return [pathtmp % s[:2]] + convert(s[2:])

    if symb in ('H', 'He'):
        if ',' in extension:
            return tuple([ALIAS[mbas]] + convert(extension.split(',')[1]))
        else:
            return ALIAS[mbas]
    else:
        return tuple([ALIAS[mbas]] + convert(extension.split(',')[0]))

OPTIMIZE_CONTRACTION = getattr(__config__, 'gto_basis_parse_optimize', False)
def parse(string, symb=None, optimize=OPTIMIZE_CONTRACTION):
    if 'ECP' in string:
        return parse_nwchem.parse_ecp(string, symb)
    else:
        return parse_nwchem.parse(string, symb, optimize)
parse.__doc__ = parse_nwchem.parse.__doc__

def parse_ecp(string, symb=None):
    # TODO: catch KeyError and provide suggestion for the possible keys
    return parse_nwchem.parse_ecp(string, symb)
parse_ecp.__doc__ = parse_nwchem.parse_ecp.__doc__

def _convert_contraction(contr_string):
    '''Parse contraction scheme string into a list

    Args:
        contr_string : str
            Desired contraction scheme in conventional 'XsYpZd...' form,
            where X, Y, Z are the total numbers of corresponding functions.
    '''
    import re
    l_fun={'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    l_fun.update({chr(num): num-100 for num in range(ord('k'), ord('p'))})
    num_contr = [int(digit) for digit in re.findall(r'\d+', contr_string)]
    basis_labels = re.findall(r'[d-z]+', contr_string)
    assert len(num_contr)==len(basis_labels)
    basis_l = [l_fun[basis] for basis in basis_labels]
    assert basis_l == sorted(basis_l), 'Contraction scheme ' + contr_string +\
        ' has to be ordered by l'
    assert len(basis_l) == len(set(basis_l)), 'Some of l in ' + contr_string +\
        ' appears more than once'
    # Prepare zero list to ensure the total length is equal to the highest l+1
    contraction_list = [0] * (1+max(basis_l))
    for (l, n_contr) in zip(basis_l, num_contr):
        contraction_list[l] = n_contr
    return contraction_list

def _truncate(basis, contr_scheme, symb, split_name):
    # keep only first n_keep contractions for each l
    contr_b = []
    b_index = 0
    for l, n_keep in enumerate(contr_scheme):
        n_saved = 0
        if n_keep > 0:
            for segm in basis:
                segm_l = segm[0]
                if segm_l == l:
                    segm_len = len(segm[1][1:])
                    n_save = min(segm_len, n_keep - n_saved)
                    if n_save > 0:
                        save_segm = [line[:n_save+1] for line in
                                     segm[:][1:]]
                        contr_b.append([l] + save_segm)
                        n_saved += n_save
            assert n_saved == n_keep, ("@{} implies {} l={} function(s), but" +
                                       "only {} in {}:{}").format(split_name[1],
                                                                  contr_scheme[l],
                                                                  l, n_saved, symb,
                                                                  split_name[0])
    return contr_b

optimize_contraction = parse_nwchem.optimize_contraction
to_general_contraction = parse_nwchem.to_general_contraction


def load(filename_or_basisname, symb, optimize=OPTIMIZE_CONTRACTION):
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
    symb = ''.join([i for i in symb if i.isalpha()])
    if '@' in filename_or_basisname:
        split_name = filename_or_basisname.split('@')
        assert len(split_name) == 2
        filename_or_basisname = split_name[0]
        contr_scheme = _convert_contraction(split_name[1].lower())
    else:
        contr_scheme = 'Full'
    if os.path.isfile(filename_or_basisname):
        # read basis from given file
        try:
            b = parse_nwchem.load(filename_or_basisname, symb, optimize)
        except BasisNotFoundError:
            with open(filename_or_basisname, 'r') as fin:
                b =  parse_nwchem.parse(fin.read(), symb)
        if contr_scheme != 'Full':
            b = _truncate(b, contr_scheme, symb, split_name)
        return b

    name = _format_basis_name(filename_or_basisname)

    if not (name in ALIAS or _is_pople_basis(name)):
        try:
            return parse_nwchem.parse(filename_or_basisname, symb)
        except IndexError:
            raise BasisNotFoundError(filename_or_basisname)
        except BasisNotFoundError as basis_err:
            pass

        try:
            return parse_nwchem.parse(filename_or_basisname)
        except IndexError:
            raise BasisNotFoundError('Invalid basis name %s' % filename_or_basisname)
        except BasisNotFoundError:
            pass

        # Last, a trial to access Basis Set Exchange database
        from pyscf.basis import bse
        if bse.basis_set_exchange is not None:
            try:
                bse_obj = bse.basis_set_exchange.api.get_basis(
                    filename_or_basisname, elements=symb)
            except KeyError:
                raise BasisNotFoundError(filename_or_basisname)
            else:
                return bse._orbital_basis(bse_obj)[0]

        raise basis_err

    if name in ALIAS:
        basmod = ALIAS[name]
    elif _is_pople_basis(name):
        basmod = _parse_pople_basis(name, symb)
    else:
        raise BasisNotFoundError(filename_or_basisname)

    if 'dat' in basmod:
        b = parse_nwchem.load(join(_BASIS_DIR, basmod), symb, optimize)
    elif isinstance(basmod, (tuple, list)) and isinstance(basmod[0], str):
        b = []
        for f in basmod:
            b += parse_nwchem.load(join(_BASIS_DIR, f), symb, optimize)
    else:
        if sys.version_info < (2,7):
            fp, pathname, description = imp.find_module(basmod, __path__)
            mod = imp.load_module(name, fp, pathname, description)
            b = mod.__getattribute__(symb)
            fp.close()
        else:
            mod = importlib.import_module('.'+basmod, __package__)
            b = mod.__getattribute__(symb)

    if contr_scheme != 'Full':
        b = _truncate(b, contr_scheme, symb, split_name)
    return b

def load_ecp(filename_or_basisname, symb):
    '''Convert the basis of the given symbol to internal format
    '''
    symb = ''.join([i for i in symb if i.isalpha()])
    if os.path.isfile(filename_or_basisname):
        # read basis from given file
        try:
            return parse_nwchem.load_ecp(filename_or_basisname, symb)
        except BasisNotFoundError:
            with open(filename_or_basisname, 'r') as fin:
                return parse_ecp(fin.read(), symb)

    name = _format_basis_name(filename_or_basisname)

    if name in ALIAS:
        basmod = ALIAS[name]
        return parse_nwchem.load_ecp(join(_BASIS_DIR, basmod), symb)

    try:
        return parse_ecp(filename_or_basisname, symb)
    except IndexError:
        raise BasisNotFoundError(filename_or_basisname)
    except BasisNotFoundError as basis_err:
        pass

    try:
        return parse_nwchem.parse_ecp(filename_or_basisname)
    except IndexError:
        raise BasisNotFoundError('Invalid basis name %s' % filename_or_basisname)
    except BasisNotFoundError:
        pass

    # Last, a trial to access Basis Set Exchange database
    from pyscf.basis import bse
    if bse.basis_set_exchange is not None:
        try:
            bse_obj = bse.basis_set_exchange.api.get_basis(
                filename_or_basisname, elements=symb)
        except KeyError:
            raise BasisNotFoundError(filename_or_basisname)
        else:
            return bse._ecp_basis(bse_obj)[0]

    raise basis_err

def _format_basis_name(basisname):
    return basisname.lower().replace('-', '').replace('_', '').replace(' ', '')

del(OPTIMIZE_CONTRACTION)
