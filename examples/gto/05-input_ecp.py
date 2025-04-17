#!/usr/bin/env python
from pyscf import gto, scf

'''
Use gto.basis.parse_ecp and gto.basis.load_ecp functions to input
user-specified ecp functions.
'''


mol = gto.M(atom='''
 Na 0. 0. 0.
 H  0.  0.  1.''',
            basis={'Na':'lanl2dz', 'H':'sto3g'},
            ecp = {'Na':'lanl2dz'})


mol = gto.M(atom='''
 Na 0. 0. 0.
 H  0.  0.  1.''',
            basis={'Na':'lanl2dz', 'H':'sto3g'},
            ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
0      2.0000000              6.0000000
1    175.5502590            -10.0000000
2      2.3365719             -6.0637782
2      0.7799867             -0.7299393
Na S
0    243.3605846              3.0000000
1     41.5764759             36.2847626
2     13.2649167             72.9304880
2      0.9764209              6.0123861
Na P
0   1257.2650682              5.0000000
1    189.6248810            117.4495683
2     54.5247759            423.3986704
2      0.9461106              7.1241813
''')})

#
# Simple arithmetic expressions can be specified in ecp input
#
mol = gto.M(atom='''
 Na 0. 0. 0.
 H  0.  0.  1.''',
            basis={'Na':'lanl2dz', 'H':'sto3g'},
            ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
0      2.0000000              6.0000000*np.exp(.5)
1    175.5502590            -10.0000000*np.exp(.5)
2      2.3365719             -6.0637782*np.exp(.5)
2      0.7799867             -0.7299393*np.exp(.5)
Na S
0    243.3605846              3.0000000
1     41.5764759             36.2847626
2     13.2649167             72.9304880
2      0.9764209              6.0123861
Na P
0   1257.2650682              5.0000000
1    189.6248810            117.4495683
2     54.5247759            423.3986704
2      0.9461106              7.1241813
''')})


#
# Burkatzki-Filippi-Dolg pseudo potential and basis are prefixed with "bfd"
#
mol = gto.M(atom='Na 0. 0. 0.; H 0 0 2.',
            basis={'Na':'bfd-vtz', 'H':'ccpvdz'},
            ecp = {'Na':'bfd-pp'})



#
# Input SOC-ECP parameters
# See also relevant introductions in
#  https://people.clarkson.edu/~pchristi/reps.html
#  http://www.nwchem-sw.org/index.php/ECP
#
# Note the SOC factor 2/(2l+1) has been multiplied in the SO coefficients
#
mol = gto.M(atom='Cu 0. 0. 0.; H 0 0 2.',
            basis={'Cu':'crenbl', 'H':'ccpvdz'},
            ecp = {'Cu': '''
#    exp                     AREP coef              SO coef
Cu nelec 10
Cu ul
2     16.30159950            -4.73227501             .063509
2     49.98759842           -34.06355667             .272004
2    173.02969360           -90.69224548            1.908336
1    651.10559082           -10.26460838            1.591675
Cu S
2      3.70869994           -87.13957977
2      4.51280022           209.05120850
2      5.53380013          -202.30523682
2     10.20059967           154.84190369
1      2.66059995             9.21743488
0     32.17929840             3.18838096
Cu P
2      3.69499993           -19.07518959            -.604602
2      4.45380020            63.05695343            2.245388
2      6.17630005          -127.18070221           -5.454358
2      8.83930016           158.41213989            8.174906
1     14.67029953            -5.66128206           -1.483716
0     30.43350029             5.39882612             .073914
'''})


#
# Input ECP and basis set from basis set exchange
#
mol = gto.M(atom='O 0. 0. 0.; S 0 0 2.',
            basis='Grimme vDZP',
            ecp = 'Grimme vDZP')
