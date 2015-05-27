#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import ctypes
import _ctypes
import pyscf.lib

_cint = pyscf.lib.load_library('libcvhf')
_cint.CINTcgto_cart.restype = ctypes.c_int
_cint.CINTcgto_spheric.restype = ctypes.c_int
_cint.CINTcgto_spinor.restype = ctypes.c_int

def getints(intor_name, atm, bas, env, bras=None, kets=None, comp=1, hermi=0):
    r'''One electron integral generator.

    Args:
        intor_name : str
            Name of the 1-electron integral.  The list of 1e integrals in
            current version of libcint (v2.5.1)

            ==========================  =========  =============
            Function                    type       Expression
            ==========================  =========  =============
            "cint1e_ovlp_sph"           spheric    ( \| \)
            "cint1e_nuc_sph"            spheric    ( \| nuc \| \)
            "cint1e_kin_sph"            spheric    (.5 \| p dot p\)
            "cint1e_ia01p_sph"          spheric    (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_sph"     spheric    (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_sph"        spheric    (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_sph"   spheric    (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_sph"     spheric    (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_sph"          spheric    (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_sph"          spheric    (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_sph"         spheric    (#C(0 1) g \|\)
            "cint1e_ignuc_sph"          spheric    (#C(0 1) g \| nuc \|\)
            "cint1e_z_sph"              spheric    ( \| zc \| \)
            "cint1e_zz_sph"             spheric    ( \| zc zc \| \)
            "cint1e_r_sph"              spheric    ( \| rc \| \)
            "cint1e_r2_sph"             spheric    ( \| rc dot rc \| \)
            "cint1e_rr_sph"             spheric    ( \| rc rc \| \)
            "cint1e_pnucp_sph"          spheric    (p* \| nuc dot p \| \)
            "cint1e_prinvxp_sph"        spheric    (p* \| rinv cross p \| \)
            "cint1e_ovlp"               spinor     ( \| \)
            "cint1e_nuc"                spinor     ( \| nuc \|\)
            "cint1e_srsr"               spinor     (sigma dot r \| sigma dot r\)
            "cint1e_sr"                 spinor     (sigma dot r \|\)
            "cint1e_srsp"               spinor     (sigma dot r \| sigma dot p\)
            "cint1e_spsp"               spinor     (sigma dot p \| sigma dot p\)
            "cint1e_sp"                 spinor     (sigma dot p \|\)
            "cint1e_spnucsp"            spinor     (sigma dot p \| nuc \| sigma dot p\)
            "cint1e_srnucsr"            spinor     (sigma dot r \| nuc \| sigma dot r\)
            "cint1e_govlp"              spinor     (g \|\)
            "cint1e_gnuc"               spinor     (g \| nuc \|\)
            "cint1e_cg_sa10sa01"        spinor     (.5 sigma cross rc \| sigma cross nabla-rinv \|\)
            "cint1e_cg_sa10sp"          spinor     (.5 rc cross sigma \| sigma dot p\)
            "cint1e_cg_sa10nucsp"       spinor     (.5 rc cross sigma \| nuc \| sigma dot p\)
            "cint1e_giao_sa10sa01"      spinor     (.5 sigma cross r \| sigma cross nabla-rinv \|\)
            "cint1e_giao_sa10sp"        spinor     (.5 r cross sigma \| sigma dot p\)
            "cint1e_giao_sa10nucsp"     spinor     (.5 r cross sigma \| nuc \| sigma dot p\)
            "cint1e_sa01sp"             spinor     (\| nabla-rinv cross sigma \| sigma dot p\)
            "cint1e_spgsp"              spinor     (g sigma dot p \| sigma dot p\)
            "cint1e_spgnucsp"           spinor     (g sigma dot p \| nuc \| sigma dot p\)
            "cint1e_spgsa01"            spinor     (g sigma dot p \| nabla-rinv cross sigma \|\)
            "cint1e_spspsp"             spinor     (sigma dot p \| sigma dot p sigma dot p\)
            "cint1e_spnuc"              spinor     (sigma dot p \| nuc \|\)
            "cint1e_ovlp_cart"          cartesian  ( \| \)
            "cint1e_nuc_cart"           cartesian  ( \| nuc \| \)
            "cint1e_kin_cart"           cartesian  (.5 \| p dot p\)
            "cint1e_ia01p_cart"         cartesian  (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_cart"    cartesian  (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_cart"       cartesian  (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_cart"  cartesian  (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_cart"    cartesian  (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_cart"         cartesian  (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_cart"         cartesian  (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_cart"        cartesian  (#C(0 1) g \|\)
            "cint1e_ignuc_cart"         cartesian  (#C(0 1) g \| nuc \|\)
            "cint1e_ipovlp_sph"         spheric    (nabla \|\)
            "cint1e_ipkin_sph"          spheric    (.5 nabla \| p dot p\)
            "cint1e_ipnuc_sph"          spheric    (nabla \| nuc \|\)
            "cint1e_iprinv_sph"         spheric    (nabla \| rinv \|\)
            "cint1e_rinv_sph"           spheric    (\| rinv \|\)
            "cint1e_ipovlp"             spinor     (nabla \|\)
            "cint1e_ipkin"              spinor     (.5 nabla \| p dot p\)
            "cint1e_ipnuc"              spinor     (nabla \| nuc \|\)
            "cint1e_iprinv"             spinor     (nabla \| rinv \|\)
            "cint1e_ipspnucsp"          spinor     (nabla sigma dot p \| nuc \| sigma dot p\)
            "cint1e_ipsprinvsp"         spinor     (nabla sigma dot p \| rinv \| sigma dot p\)
            "cint1e_ipovlp_cart"        cartesian  (nabla \|\)
            "cint1e_ipkin_cart"         cartesian  (.5 nabla \| p dot p\)
            "cint1e_ipnuc_cart"         cartesian  (nabla \| nuc \|\)
            "cint1e_iprinv_cart"        cartesian  (nabla \| rinv \|\)
            "cint1e_rinv_cart"          cartesian  (\| rinv \|\)
            ==========================  =========  =============

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

    Kwargs:
        bras : list of int
            shell ids for bra.  Default is all shells given by bas
        kets : list of int
            shell ids for ket.  Default is all shells given by bas
        comp : int
            Components of the integrals, e.g. cint1e_ipovlp has 3 components.
        hermi : int
            Symmetry of the integrals
            | 0 : no symmetry assumed (default)
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

    Examples:

    >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
    >>> gto.moleintor('cint1e_ipnuc_sph', mol._atm, mol._bas, mol._env, comp=3) # <nabla i | V_nuc | j>
    [[[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.10289944  0.48176097]
      [-0.48176097 -0.10289944]]]
    '''
    nbas = len(bas)
    if bras is None:
        bras = range(nbas)
    else:
        assert(max(bras) < len(bas))
    if kets is None:
        kets = range(nbas)
    else:
        assert(max(kets) < len(bas))

    c_atm = atm.ctypes.data_as(pyscf.lib.c_int_p)
    c_bas = bas.ctypes.data_as(pyscf.lib.c_int_p)
    c_env = env.ctypes.data_as(pyscf.lib.c_double_p)
    c_natm = ctypes.c_int(atm.shape[0])
    c_nbas = ctypes.c_int(bas.shape[0])
    nbra = len(bras)
    nket = len(kets)

    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgto_of = _cint.CINTcgto_cart
        c_intor = _cint.GTO1eintor_cart
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgto_of = _cint.CINTcgto_spheric
        c_intor = _cint.GTO1eintor_sph
    else:
        dtype = numpy.complex
        num_cgto_of = _cint.CINTcgto_spinor
        c_intor = _cint.GTO1eintor_spinor
    naoi = sum([num_cgto_of(ctypes.c_int(i), c_bas) for i in bras])
    naoj = sum([num_cgto_of(ctypes.c_int(i), c_bas) for i in kets])

    bralst = numpy.array(bras, dtype=numpy.int32)
    ketlst = numpy.array(kets, dtype=numpy.int32)
    mat = numpy.empty((comp,naoi,naoj), dtype)
    fnaddr = ctypes.c_void_p(_ctypes.dlsym(_cint._handle, intor_name))
    c_intor(fnaddr, mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp), \
            ctypes.c_int(hermi), \
            bralst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbra),
            ketlst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nket),
            c_atm, c_natm, c_bas, c_nbas, c_env)

    if comp == 1:
        mat = mat.reshape(naoi,naoj)
    if hermi == 0:
        return mat
    else:
        if comp == 1:
            pyscf.lib.hermi_triu(mat, hermi=hermi)
        else:
            for i in range(comp):
                pyscf.lib.hermi_triu(mat[i], hermi=hermi)
        return mat

def getints_by_shell(intor_name, shls, atm, bas, env, comp=1):
    r'''For given 2, 3 or 4 shells, interface for libcint to get 1e, 2e,
    2-center-2e or 3-center-2e integrals

    Args:
        intor_name : str
            Integral name.  In the current version of libcint (v2.5.1), it can be

            ==========================  =========  =============
            Function                    type       Expression
            ==========================  =========  =============
            "cint1e_ovlp_sph"           spheric    ( \| \)
            "cint1e_nuc_sph"            spheric    ( \| nuc \| \)
            "cint1e_kin_sph"            spheric    (.5 \| p dot p\)
            "cint1e_ia01p_sph"          spheric    (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_sph"     spheric    (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_sph"        spheric    (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_sph"   spheric    (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_sph"     spheric    (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_sph"          spheric    (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_sph"          spheric    (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_sph"         spheric    (#C(0 1) g \|\)
            "cint1e_ignuc_sph"          spheric    (#C(0 1) g \| nuc \|\)
            "cint1e_z_sph"              spheric    ( \| zc \| \)
            "cint1e_zz_sph"             spheric    ( \| zc zc \| \)
            "cint1e_r_sph"              spheric    ( \| rc \| \)
            "cint1e_r2_sph"             spheric    ( \| rc dot rc \| \)
            "cint1e_rr_sph"             spheric    ( \| rc rc \| \)
            "cint1e_pnucp_sph"          spheric    (p* \| nuc dot p \| \)
            "cint1e_prinvxp_sph"        spheric    (p* \| rinv cross p \| \)
            "cint1e_ovlp"               spinor     ( \| \)
            "cint1e_nuc"                spinor     ( \| nuc \|\)
            "cint1e_srsr"               spinor     (sigma dot r \| sigma dot r\)
            "cint1e_sr"                 spinor     (sigma dot r \|\)
            "cint1e_srsp"               spinor     (sigma dot r \| sigma dot p\)
            "cint1e_spsp"               spinor     (sigma dot p \| sigma dot p\)
            "cint1e_sp"                 spinor     (sigma dot p \|\)
            "cint1e_spnucsp"            spinor     (sigma dot p \| nuc \| sigma dot p\)
            "cint1e_srnucsr"            spinor     (sigma dot r \| nuc \| sigma dot r\)
            "cint1e_govlp"              spinor     (g \|\)
            "cint1e_gnuc"               spinor     (g \| nuc \|\)
            "cint1e_cg_sa10sa01"        spinor     (.5 sigma cross rc \| sigma cross nabla-rinv \|\)
            "cint1e_cg_sa10sp"          spinor     (.5 rc cross sigma \| sigma dot p\)
            "cint1e_cg_sa10nucsp"       spinor     (.5 rc cross sigma \| nuc \| sigma dot p\)
            "cint1e_giao_sa10sa01"      spinor     (.5 sigma cross r \| sigma cross nabla-rinv \|\)
            "cint1e_giao_sa10sp"        spinor     (.5 r cross sigma \| sigma dot p\)
            "cint1e_giao_sa10nucsp"     spinor     (.5 r cross sigma \| nuc \| sigma dot p\)
            "cint1e_sa01sp"             spinor     (\| nabla-rinv cross sigma \| sigma dot p\)
            "cint1e_spgsp"              spinor     (g sigma dot p \| sigma dot p\)
            "cint1e_spgnucsp"           spinor     (g sigma dot p \| nuc \| sigma dot p\)
            "cint1e_spgsa01"            spinor     (g sigma dot p \| nabla-rinv cross sigma \|\)
            "cint1e_spspsp"             spinor     (sigma dot p \| sigma dot p sigma dot p\)
            "cint1e_spnuc"              spinor     (sigma dot p \| nuc \|\)
            "cint1e_ovlp_cart"          cartesian  ( \| \)
            "cint1e_nuc_cart"           cartesian  ( \| nuc \| \)
            "cint1e_kin_cart"           cartesian  (.5 \| p dot p\)
            "cint1e_ia01p_cart"         cartesian  (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_cart"    cartesian  (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_cart"       cartesian  (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_cart"  cartesian  (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_cart"    cartesian  (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_cart"         cartesian  (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_cart"         cartesian  (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_cart"        cartesian  (#C(0 1) g \|\)
            "cint1e_ignuc_cart"         cartesian  (#C(0 1) g \| nuc \|\)
            "cint1e_ipovlp_sph"         spheric    (nabla \|\)
            "cint1e_ipkin_sph"          spheric    (.5 nabla \| p dot p\)
            "cint1e_ipnuc_sph"          spheric    (nabla \| nuc \|\)
            "cint1e_iprinv_sph"         spheric    (nabla \| rinv \|\)
            "cint1e_rinv_sph"           spheric    (\| rinv \|\)
            "cint1e_ipovlp"             spinor     (nabla \|\)
            "cint1e_ipkin"              spinor     (.5 nabla \| p dot p\)
            "cint1e_ipnuc"              spinor     (nabla \| nuc \|\)
            "cint1e_iprinv"             spinor     (nabla \| rinv \|\)
            "cint1e_ipspnucsp"          spinor     (nabla sigma dot p \| nuc \| sigma dot p\)
            "cint1e_ipsprinvsp"         spinor     (nabla sigma dot p \| rinv \| sigma dot p\)
            "cint1e_ipovlp_cart"        cartesian  (nabla \|\)
            "cint1e_ipkin_cart"         cartesian  (.5 nabla \| p dot p\)
            "cint1e_ipnuc_cart"         cartesian  (nabla \| nuc \|\)
            "cint1e_iprinv_cart"        cartesian  (nabla \| rinv \|\)
            "cint1e_rinv_cart"          cartesian  (\| rinv \|\)
            "cint2e_p1vxp1_sph"         spheric    ( p* \, cross p \| \, \) ; SSO
            "cint2e_sph"                spheric    ( \, \| \, \)
            "cint2e_ig1_sph"            spheric    (#C(0 1) g \, \| \, \)
            "cint2e"                    spinor     (, \| \, \)
            "cint2e_spsp1"              spinor     (sigma dot p \, sigma dot p \| \, \)
            "cint2e_spsp1spsp2"         spinor     (sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p \)
            "cint2e_srsr1"              spinor     (sigma dot r \, sigma dot r \| \,\)
            "cint2e_srsr1srsr2"         spinor     (sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "cint2e_cg_sa10sp1"         spinor     (.5 rc cross sigma \, sigma dot p \| \,\)
            "cint2e_cg_sa10sp1spsp2"    spinor     (.5 rc cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "cint2e_giao_sa10sp1"       spinor     (.5 r cross sigma \, sigma dot p \| \,\)
            "cint2e_giao_sa10sp1spsp2"  spinor     (.5 r cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "cint2e_g1"                 spinor     (g \, \| \,\)
            "cint2e_spgsp1"             spinor     (g sigma dot p \, sigma dot p \| \,\)
            "cint2e_g1spsp2"            spinor     (g \, \| sigma dot p \, sigma dot p\)
            "cint2e_spgsp1spsp2"        spinor     (g sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "cint2e_spv1"               spinor     (sigma dot p \, \| \,\)
            "cint2e_vsp1"               spinor     (\, sigma dot p \| \,\)
            "cint2e_spsp2"              spinor     (\, \| sigma dot p \, sigma dot p\)
            "cint2e_spv1spv2"           spinor     (sigma dot p \, \| sigma dot p \,\)
            "cint2e_vsp1spv2"           spinor     (\, sigma dot p \| sigma dot p \,\)
            "cint2e_spv1vsp2"           spinor     (sigma dot p \, \| \, sigma dot p\)
            "cint2e_vsp1vsp2"           spinor     (\, sigma dot p \| \, sigma dot p\)
            "cint2e_spv1spsp2"          spinor     (sigma dot p \, \| sigma dot p \, sigma dot p\)
            "cint2e_vsp1spsp2"          spinor     (\, sigma dot p \| sigma dot p \, sigma dot p\)
            "cint2e_ig1_cart"           cartesian  (#C(0 1) g \, \| \, \)
            "cint2e_ip1_sph"            spheric    (nabla \, \| \,\)
            "cint2e_ip1"                spinor     (nabla \, \| \,\)
            "cint2e_ipspsp1"            spinor     (nabla sigma dot p \, sigma dot p \| \,\)
            "cint2e_ip1spsp2"           spinor     (nabla \, \| sigma dot p \, sigma dot p\)
            "cint2e_ipspsp1spsp2"       spinor     (nabla sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "cint2e_ipsrsr1"            spinor     (nabla sigma dot r \, sigma dot r \| \,\)
            "cint2e_ip1srsr2"           spinor     (nabla \, \| sigma dot r \, sigma dot r\)
            "cint2e_ipsrsr1srsr2"       spinor     (nabla sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "cint2e_ip1_cart"           cartesian  (nabla \, \| \,\)
            "cint2e_ssp1ssp2"           spinor     ( \, sigma dot p \| gaunt \| \, sigma dot p\)
            "cint2e_cg_ssa10ssp2"       spinor     (rc cross sigma \, \| gaunt \| \, sigma dot p\)
            "cint2e_giao_ssa10ssp2"     spinor     (r cross sigma  \, \| gaunt \| \, sigma dot p\)
            "cint2e_gssp1ssp2"          spinor     (g \, sigma dot p  \| gaunt \| \, sigma dot p\)
            "cint2e_ipip1_sph"          spheric    ( nabla nabla \, \| \, \)
            "cint2e_ipvip1_sph"         spheric    ( nabla \, nabla \| \, \)
            "cint2e_ip1ip2_sph"         spheric    ( nabla \, \| nabla \, \)
            "cint3c2e_ip1_sph"          spheric    (nabla \, \| \)
            "cint3c2e_ip2_sph"          spheric    ( \, \| nabla\)
            "cint2c2e_ip1_sph"          spheric    (nabla \| r12 \| \)
            "cint3c2e_spinor"           spinor     (nabla \, \| \)
            "cint3c2e_spsp1_spinor"     spinor     (nabla \, \| \)
            "cint3c2e_ip1_spinor"       spinor     (nabla \, \| \)
            "cint3c2e_ip2_spinor"       spinor     ( \, \| nabla\)
            "cint3c2e_ipspsp1_spinor"   spinor     (nabla sigma dot p \, sigma dot p \| \)
            "cint3c2e_spsp1ip2_spinor"  spinor     (sigma dot p \, sigma dot p \| nabla \)
            ==========================  =========  =============

        shls : list of int
            The AO shell-ids of the integrals
        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

    Kwargs:
        comp : int
            Components of the integrals, e.g. cint1e_ipovlp has 3 components.

    Returns:
        ndarray of 2-dim to 5-dim, depending on the integral type (1e,
        2e, 3c-2e, 2c2e) and the value of comp

    Examples:
        The gradients of the spherical 2e integrals

    >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
    >>> gto.getints_by_shell('cint2e_ip1_sph', (0,1,0,1), mol._atm, mol._bas, mol._env, comp=3)
    [[[[[-0.        ]]]]
      [[[[-0.        ]]]]
      [[[[-0.08760462]]]]]
    '''
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    natm = ctypes.c_int(atm.shape[0])
    nbas = ctypes.c_int(bas.shape[0])
    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgto_of = lambda basid: _cint.CINTcgto_cart(ctypes.c_int(basid),
                                                        c_bas)
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgto_of = lambda basid: _cint.CINTcgto_spheric(ctypes.c_int(basid),
                                                           c_bas)
    else:
        dtype = numpy.complex
        num_cgto_of = lambda basid: _cint.CINTcgto_spinor(ctypes.c_int(basid),
                                                          c_bas)
    if '3c2e' in intor_name or '2e3c' in intor_name:
        assert(len(shls) == 3)
        #di, dj, dk = map(num_cgto_of, shls)
        di = num_cgto_of(shls[0])
        dj = num_cgto_of(shls[1])
        dk = _cint.CINTcgto_spheric(ctypes.c_int(shls[2]), c_bas) # spheric-GTO for aux function?
        buf = numpy.empty((di,dj,dk,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*3)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj,dk)
        else:
            return buf.transpose(3,0,1,2)
    elif '2c2e' in intor_name or '2e2c' in intor_name:
        assert(len(shls) == 2)
        #di, dj = map(num_cgto_of, shls)
        #buf = numpy.empty((di,dj,comp), dtype, order='F')
        di = _cint.CINTcgto_spheric(ctypes.c_int(shls[0]), c_bas)
        dj = _cint.CINTcgto_spheric(ctypes.c_int(shls[1]), c_bas)
        buf = numpy.empty((di,dj,comp), order='F') # no complex?
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)
    elif '2e' in intor_name:
        assert(len(shls) == 4)
        di, dj, dk, dl = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,dk,dl,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*4)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj,dk,dl)
        else:
            return buf.transpose(4,0,1,2,3)
    else:
        assert(len(shls) == 2)
        di, dj = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p))
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        ["H", (0,  0, 0  )],
        ["H", (0,  0, 1  )],
    ])
    mol.basis = {"H": 'cc-pvdz'}
    mol.build()
    mol.set_rinv_origin_(mol.atom_coord(0))
    for i in range(mol.nbas):
        for j in range(mol.nbas):
            print(i, j, getints_by_shell('cint1e_prinvxp_sph', (i,j),
                                         mol._atm, mol._bas, mol._env, 3))
