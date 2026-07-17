#!/usr/bin/env python

'''
User defined XC functional

Two approaches are demonstrated:
1. parse_xc: string-based functional composition (combines existing LibXC
   functionals but cannot modify their internal parameters)
2. register_custom_functional_: modify a LibXC functional's internal
   parameters and register it under a custom name

See also
* pyscf.dft.libxc for implementation details
* Example 24-define_xc_functional.py to define a completely new functional
'''

import numpy
from pyscf import gto
from pyscf import dft

mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587 ''',
    basis = 'ccpvdz')

#
# ---- parse_xc: string-based XC functional composition ----
#
# DFT can parse the custom XC functional, following the rules:
# * The given functional description must be a one-line string.
# * The functional description is case-insensitive.
# * The functional description string has two parts, separated by ",".  The
#   first part describes the exchange functional, the second is the correlation
#   functional.
#   - If "," was not presented in string, the entire string is treated as a
#     compound XC functional (including both X and C functionals, such as b3lyp).
#   - To input only X functional (without C functional), leave the second part
#     blank. E.g. description='slater,' for pure LDA functional.
#   - To neglect the X functional (only input C functional), leave the first
#     part blank. E.g. description=',vwn' means pure VWN functional
#   - If compound XC functional is specified, no matter whehter it is in the X
#     part (the string in front of comma) or the C part (the string behind
#     comma), both X and C functionals of the compound XC functional will be
#     used.
# * The functional name can be placed in arbitrary order.  Two names needs to
#   be separated by operators + or -.  Blank spaces are ignored.
#   NOTE the parser only reads operators + - *.  / is not supported.
# * A functional name can have at most one factor.  If the factor is not
#   given, it is set to 1.  Compound functional can be scaled as a unit. For
#   example '0.5*b3lyp' is equivalent to
#   'HF*0.1 + .04*LDA + .36*B88, .405*LYP + .095*VWN'
# * String "HF" stands for exact exchange (HF K matrix).  It is allowed to
#   put "HF" in C (correlation) functional part.
# * String "RSH" means range-separated operator. Its format is
#   RSH(omega, alpha, beta).  Another way to input RSH is to use keywords
#   SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
#   alpha" where the number in parenthesis is the value of omega.
# * Be careful with the libxc convention on GGA functional, in which the LDA
#   contribution has been included.

mf = dft.RKS(mol)
# B3LYP can be constructed
mf.xc = 'HF*0.2 + .08*LDA + .72*B88, .81*LYP + .19*VWN'
e1 = mf.kernel()
print('E = %.15g  ref = -76.3832244350081' % e1)

#
# No correlation functional
#
mf.xc = '.2*HF + .08*LDA + .72*B88'
e1 = mf.kernel()
print('E = %.15g  ref = -75.9807850596666' % e1)

#
# If not given, the factor for each functional equals 1 by default.
#
mf = dft.RKS(mol)
mf.xc = 'b88,lyp'
e1 = mf.kernel()

mf = dft.RKS(mol)
mf.xc = 'b88*1,lyp*1'
eref = mf.kernel()
print('%.15g == %.15g' % (e1, eref))

#
# Compound functionals can be used as part of the definition
#
mf = dft.RKS(mol)
mf.xc = '0.5*b3lyp, 0.5*lyp'
e1 = mf.kernel()
print('E = %.15g  ref = -71.9508340443282' % e1)

# Compound XC functional can be presented in the C part (the string behind
# comma). Both X and C functionals of the compound XC functional will be used.
# Compound XC functional can be scaled as a unit.
#
mf = dft.RKS(mol)
mf.xc = '0.5*b3lyp, 0.5*b3p86'
e1 = mf.kernel()

mf = dft.RKS(mol)
mf.xc = '0.5*b3lyp + 0.5*b3p86'
e2 = mf.kernel()
print('E1 = %.15g  E2 = %.15g  ref = -76.3923625924023' % (e1, e2))

#
# More examples of customized functionals. NOTE These customized functionals
# are presented for the purpose of demonstrating the feature of the XC input.
# They are not reported in any literature. DO NOT use them in the actual
# calculations.
#
# Half HF exchange plus half B3LYP plus half VWN functional
mf.xc = '.5*HF+.5*B3LYP,VWN*.5'

# "-" to subtract one functional from another
mf.xc = 'B88 - SLATER*.5'

# The functional below gives omega = 0.33, alpha = 0.6 * 0.65 = 0.39
# beta = -0.46 * 0.6 + 0.4 * 0.2(from HF of B3P86) = -0.196
mf.xc = '0.6*CAM_B3LYP+0.4*B3P86'

# The input XC description does not depend on the order of functionals. The
# functional below is the same to the functional above
mf.xc = '0.4*B3P86+0.6*CAM_B3LYP'

# Use SR_HF/LR_HF keywords to input range-separated functionals.
# When exact HF exchange is presented, it is split into SR and LR parts
# alpha = 0.8(from HF) + 0.22
# beta = 0.5 + 0.8(from HF)
mf.xc = '0.5*SR-HF(0.3) + .8*HF + .22*LR_HF'

# RSH is another keyword to input range-separated functionals
mf.xc = '0.5*RSH(0.3,2.04,0.56) + 0.5*BP86'

# A shorthand to input 'PBE,PBE', which is a compound functional. Note the
# shorthand input is different to the two examples 'PBE,' and ',PBE' below.
mf.xc = 'PBE'

# Exchange part only
mf.xc = 'PBE,'

# Correlation part only
mf.xc = ',PBE'

# When defining custom functionals, compound functional will affect both the
# exchange and correlation parts
mf.xc = 'PBE + SLATER*.5'

# The above input is equivalent to
mf.xc = 'PBE + SLATER*.5, PBE'

#
# ---- register_custom_functional_: modify LibXC functional parameters ----
#
# register_custom_functional_ registers a custom functional based on an
# existing LibXC functional with modified parameters. Only the exchange
# component MGGA_X_TPSS (ID 202) is modified below; MGGA_C_TPSS is
# unmodified. MGGA_X_TPSS has 7 parameters: _b, _c, _e, _kappa, _mu,
# _BLOC_a, _BLOC_b.
#

XC_ID_TPSS_X = 202
mf = dft.RKS(mol)

#
# Example 1: Array-style ext_params.
#
param_arr = numpy.array([0.15, 0.88491, 0.047, 0.872, 0.16952, 2.0, 0.0])
dft.libxc.register_custom_functional_(
    'my-tpss-arr', 'tpss',
    ext_params={XC_ID_TPSS_X: param_arr})

mf.xc = 'my-tpss-arr'
e_arr = mf.kernel()
print('E(my-tpss-arr) = %.15g' % e_arr)

#
# Example 2: Named-dict ext_params.
#
named = {
    '_b': 0.15, '_c': 0.88491, '_e': 0.047, '_kappa': 0.872,
    '_mu': 0.16952, '_BLOC_a': 2.0, '_BLOC_b': 0.0,
}
dft.libxc.register_custom_functional_(
    'my-tpss-dict', 'tpss',
    ext_params={XC_ID_TPSS_X: named})

mf.xc = 'my-tpss-dict'
e_dict = mf.kernel()
print('E(my-tpss-dict) = %.15g  (should equal %.15g)' % (e_dict, e_arr))

#
# Example 3: Partial named-dict ext_params. Omitted keys (_BLOC_a, _BLOC_b)
# keep native defaults.
#
named_partial = {'_b': 0.15, '_c': 0.88491, '_e': 0.047,
                 '_kappa': 0.872, '_mu': 0.16952}
dft.libxc.register_custom_functional_(
    'my-tpss-partial', 'tpss',
    ext_params={XC_ID_TPSS_X: named_partial})

mf.xc = 'my-tpss-partial'
e_partial = mf.kernel()
print('E(my-tpss-partial) = %.15g  (should equal %.15g)' % (e_partial, e_arr))

#
# Compare with native TPSS.
#
mf.xc = 'tpss'
e_tpss = mf.kernel()
print('E(tpss) = %.15g' % e_tpss)

#
# unregister_custom_functional_ removes a custom functional.
#
dft.libxc.unregister_custom_functional_('my-tpss-arr')
dft.libxc.unregister_custom_functional_('my-tpss-dict')
dft.libxc.unregister_custom_functional_('my-tpss-partial')

