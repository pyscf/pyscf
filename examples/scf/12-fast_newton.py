#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
scf.fast_newton is an entry for second order SCF method with optimized
configures. It can perform 3 times faster than the default second order SCF
settings in many systems.
'''

from pyscf import gto, scf

mol = gto.M(atom='''
N 5.254640 8.787990 1.967400
Cu 6.187640 7.420980 0.928170
O 5.960990 8.763630 -1.009340
N 7.206250 6.158700 -0.128810
Cu 7.353770 6.763870 4.005480
O 5.858810 5.762100 4.667340
Cu 4.855440 5.063760 3.115860
O 6.373750 6.251140 2.444720
N 7.593860 7.496140 5.874780
N 8.269230 8.364600 3.116600
N 9.103700 5.317610 4.094400
N 4.193360 4.677270 1.233480
N 3.125440 5.049530 4.452150
N 5.398520 2.932600 3.447970
O 6.126889 3.741950 6.365440
H 5.755350 9.731720 -1.014880
H 6.393530 8.597630 -1.885180
H 6.148920 4.991940 5.235600
H 6.823489 3.795300 7.077680
H 5.987039 2.788730 6.202950
H 4.385099 8.445653 2.390932
H 5.816548 9.154453 2.743699
H 4.988101 9.605127 1.407115
H 6.771983 5.230175 -0.172724
H 7.325546 6.456228 -1.103449
H 8.153994 6.005554 0.233135
H 3.372639 5.237613 5.430001
H 2.420871 5.752619 4.203299
H 2.631033 4.150531 4.458321
H 6.225091 2.641675 2.914287
H 5.612530 2.721618 4.428971
H 4.651591 2.282146 3.180231
H 4.947415 4.665452 0.537824
H 3.725531 3.767789 1.151899
H 3.515369 5.369877 0.896878
H 9.224480 4.780740 3.228456
H 10.002053 5.784964 4.259380
H 9.010897 4.621544 4.842435
H 7.210886 6.878915 6.599377
H 8.578203 7.641391 6.125073
H 7.135300 8.403680 6.011782
H 8.308553 8.287098 2.094287
H 7.797036 9.254716 3.309999
H 9.239940 8.492427 3.423290 
''',
            basis = 'ccpvdz',
            charge = 3,
            spin = 3,
            verbose = 4,
            output = 'cu3.out')

mf = scf.fast_newton(scf.RHF(mol))
print('E(tot) %.15g  ref = -5668.38221757799' % mf.e_tot)

#
# scf.fast_newton function can be used for specific initial guess.
#
# We first create an initial guess with DIIS iterations. The DIIS results are
# saved in a checkpoint file which can be load in another calculation.
#
mf = scf.RHF(mol)
mf.chkfile = 'cu3-diis.chk'
mf.max_cycle = 2
mf.kernel()

#
# Load the DIIS results then use it as initial guess for function
# scf.fast_newton
#
mf = scf.RHF(mol)
mf.__dict__.update(scf.chkfile.load('cu3-diis.chk', 'scf'))
scf.fast_newton(mf)
