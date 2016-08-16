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
# Burkatzki-Filippi-Dolg pseudo potential and basis are prefixed with "bfd"
#
mol = gto.M(atom='Na 0. 0. 0.; H 0 0 2.',
            basis={'Na':'bfd-vtz', 'H':'ccpvdz'},
            ecp = {'Na':'bfd-pp'})

