#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import ctypes
import pyscf.lib

libcgto = pyscf.lib.load_library('libcgto')
libcgto.CINTcgto_cart.restype = ctypes.c_int
libcgto.CINTcgto_spheric.restype = ctypes.c_int
libcgto.CINTcgto_spinor.restype = ctypes.c_int
libcvhf = pyscf.lib.load_library('libcvhf')

ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8

def getints(intor_name, atm, bas, env, shls_slice=None, comp=1, hermi=0,
            aosym='s1', ao_loc=None, cintopt=None, out=None):
    r'''1e and 2e integral generator.

    Args:
        intor_name : str

            ==========================  =========  =============
            Function                    type       Expression
            ==========================  =========  =============
            "cint1e_ovlp_sph"           spherical  ( \| \)
            "cint1e_nuc_sph"            spherical  ( \| nuc \| \)
            "cint1e_kin_sph"            spherical  (.5 \| p dot p\)
            "cint1e_ia01p_sph"          spherical  (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_sph"     spherical  (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_sph"        spherical  (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_sph"   spherical  (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_sph"     spherical  (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_sph"          spherical  (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_sph"          spherical  (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_sph"         spherical  (#C(0 1) g \|\)
            "cint1e_ignuc_sph"          spherical  (#C(0 1) g \| nuc \|\)
            "cint1e_z_sph"              spherical  ( \| zc \| \)
            "cint1e_zz_sph"             spherical  ( \| zc zc \| \)
            "cint1e_r_sph"              spherical  ( \| rc \| \)
            "cint1e_r2_sph"             spherical  ( \| rc dot rc \| \)
            "cint1e_rr_sph"             spherical  ( \| rc rc \| \)
            "cint1e_pnucp_sph"          spherical  (p* \| nuc dot p \| \)
            "cint1e_prinvxp_sph"        spherical  (p* \| rinv cross p \| \)
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
            "cint1e_ipovlp_sph"         spherical  (nabla \|\)
            "cint1e_ipkin_sph"          spherical  (.5 nabla \| p dot p\)
            "cint1e_ipnuc_sph"          spherical  (nabla \| nuc \|\)
            "cint1e_iprinv_sph"         spherical  (nabla \| rinv \|\)
            "cint1e_rinv_sph"           spherical  (\| rinv \|\)
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
            "cint2e_p1vxp1_sph"         spherical  ( p* \, cross p \| \, \) ; SSO
            "cint2e_sph"                spherical  ( \, \| \, \)
            "cint2e_ig1_sph"            spherical  (#C(0 1) g \, \| \, \)
            "cint2e_ig1_cart"           cartesian  (#C(0 1) g \, \| \, \)
            "cint2e_ip1_sph"            spherical  (nabla \, \| \,\)
            "cint2e_ip1_cart"           cartesian  (nabla \, \| \,\)
            "cint2e_ipip1_sph"          spherical  ( nabla nabla \, \| \, \)
            "cint2e_ipvip1_sph"         spherical  ( nabla \, nabla \| \, \)
            "cint2e_ip1ip2_sph"         spherical  ( nabla \, \| nabla \, \)
            "cint3c2e_ip1_sph"          spherical  (nabla \, \| \)
            "cint3c2e_ip2_sph"          spherical  ( \, \| nabla\)
            "cint2c2e_ip1_sph"          spherical  (nabla \| r12 \| \)
            ==========================  =========  =============

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

    Kwargs:
        shls_slice : 8-element list
            (ish_start, ish_end, jsh_start, jsh_end, ksh_start, ksh_end, lsh_start, lsh_end)
        comp : int
            Components of the integrals, e.g. cint1e_ipovlp has 3 components.
        hermi : int (1e integral only)
            Symmetry of the 1e integrals

            | 0 : no symmetry assumed (default)
            | 1 : hermitian
            | 2 : anti-hermitian

        aosym : str (2e integral only)
            Symmetry of the 2e integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry

        out : ndarray (2e integral only)
            array to store the 2e AO integrals

    Returns:
        ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

    Examples:

    >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
    >>> gto.getints('cint1e_ipnuc_sph', mol._atm, mol._bas, mol._env, comp=3) # <nabla i | V_nuc | j>
    [[[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.10289944  0.48176097]
      [-0.48176097 -0.10289944]]]
    '''
    if (intor_name.startswith('cint1e') or
        intor_name.startswith('ECP') or
        intor_name.startswith('cint2c2e')):
        return getints2c(intor_name, atm, bas, env, shls_slice, comp,
                         hermi, ao_loc, cintopt, out)
    elif intor_name.startswith('cint2e'):
        return getints2e(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    elif intor_name.startswith('cint3c'):
        return getints3c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    else:
        raise RuntimeError('Unknown intor %s' % intor_name)

def getints2c(intor_name, atm, bas, env, shls_slice=None, comp=1, hermi=0,
              ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    else:
        assert(shls_slice[1] <= nbas and shls_slice[3] <= nbas)
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)

    i0, i1, j0, j1 = shls_slice[:4]
    naoi = ao_loc[i1] - ao_loc[i0];
    naoj = ao_loc[j1] - ao_loc[j0];
    if '_cart' in intor_name or '_sph' in intor_name:
        mat = numpy.ndarray((naoi,naoj,comp), numpy.double, out, order='F')
        if '2c2e' in intor_name:
            drv_name = 'GTOint2c2e'
        else:
            drv_name = 'GTOint2c'
    else:
        mat = numpy.ndarray((naoi,naoj,comp), numpy.complex, out, order='F')
        if '2c2e' in intor_name:
            drv_name = 'GTOint2c2e_spinor'
            assert(hermi != pyscf.lib.HERMITIAN and
                   hermi != pyscf.lib.ANTIHERMI)
        else:
            drv_name = 'GTOint2c_spinor'

#    if cintopt is None:
#        intopt = make_cintopt(atm, bas, env, intor_name)
#    else:
#        intopt = cintopt
    intopt = pyscf.lib.c_null_ptr()

    fn = getattr(libcgto, drv_name)
    fn(getattr(libcgto, intor_name), mat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(hermi),
       (ctypes.c_int*4)(*(shls_slice[:4])),
       ao_loc.ctypes.data_as(ctypes.c_void_p), intopt,
       atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
       env.ctypes.data_as(ctypes.c_void_p))

    if comp == 1:
        return mat.reshape((naoi,naoj), order='A')
    else:
        return mat.transpose(2,0,1)

def getints3c(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)
    else:
        assert(shls_slice[1] <= nbas and
               shls_slice[3] <= nbas and
               shls_slice[5] <= nbas)
    if ao_loc is None:
        assert('ssc' not in intor_name)
        ao_loc = make_loc(bas, intor_name)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    naok = ao_loc[k1] - ao_loc[k0];

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0];
        naoj = ao_loc[j1] - ao_loc[j0];
        shape = (naoi, naoj, naok, comp)
    else:
        aosym = 's2ij'
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok, comp)

    if '_cart' in intor_name or '_sph' in intor_name or 'ssc' in intor_name:
        mat = numpy.ndarray(shape, numpy.double, out, order='F')
    else:
        raise NotImplementedError
        mat = numpy.ndarray(shape, numpy.complex, out, order='F')

    if cintopt is None:
        intopt = make_cintopt(atm, bas, env, intor_name)
    else:
        intopt = cintopt

    drv = libcgto.GTOnr3c_drv
    drv(getattr(libcgto, intor_name), getattr(libcgto, 'GTOnr3c_fill_'+aosym),
        mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
        (ctypes.c_int*6)(*(shls_slice[:6])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), intopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    if comp == 1:
        return mat.reshape(shape[:-1], order='A')
    else:
        return numpy.rollaxis(mat, -1, 0)

def getints2e(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    aosym = _stand_sym_code(aosym)

    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(pyscf.lib.c_int_p)
    c_bas = bas.ctypes.data_as(pyscf.lib.c_int_p)
    c_env = env.ctypes.data_as(pyscf.lib.c_double_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]

    if '_cart' in intor_name:
        libcgto.CINTtot_cgto_cart.restype = ctypes.c_int
        nao = libcgto.CINTtot_cgto_cart(c_bas, ctypes.c_int(nbas))
        cgto_in_shell = 'CINTcgto_cart'
    elif '_sph' in intor_name:
        libcgto.CINTtot_cgto_spheric.restype = ctypes.c_int
        nao = libcgto.CINTtot_cgto_spheric(c_bas, ctypes.c_int(nbas))
        cgto_in_shell = 'CINTcgto_spheric'
    else:
        raise NotImplementedError('cint2e spinor AO integrals')

    if intor_name in ('cint2e_sph', 'cint2e_cart') and aosym == 's8':
        assert(shls_slice is None)
        nao_pair = nao*(nao+1)//2
        if out is None:
            out = numpy.empty((nao_pair*(nao_pair+1)//2))
        else:
            out = numpy.ndarray((nao_pair*(nao_pair+1)//2), buffer=out)
        drv = libcvhf.GTO2e_cart_or_sph
        drv(getattr(libcgto, intor_name), getattr(libcgto, cgto_in_shell),
            out.ctypes.data_as(ctypes.c_void_p),
            c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
        return out

    else:
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
        elif len(shls_slice) == 4:
            shls_slice = shls_slice + (0, nbas, 0, nbas)
        else:
            assert(shls_slice[1] <= nbas and shls_slice[3] <= nbas and
                   shls_slice[5] <= nbas and shls_slice[7] <= nbas)
        ao_loc = make_loc(bas, intor_name)
        i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        naok = ao_loc[k1] - ao_loc[k0]
        naol = ao_loc[l1] - ao_loc[l0]
        if aosym in ('s4', 's2ij'):
            nij = naoi * (naoi + 1) // 2
            assert(numpy.all(ao_loc[i0:i1]-ao_loc[i0] == ao_loc[j0:j1]-ao_loc[j0]))
        else:
            nij = naoi * naoj
        if aosym in ('s4', 's2kl'):
            nkl = naok * (naok + 1) // 2
            assert(numpy.all(ao_loc[k0:k1]-ao_loc[k0] == ao_loc[l0:l1]-ao_loc[l0]))
        else:
            nkl = naok * naol
        if comp == 1:
            out = numpy.ndarray((nij,nkl), buffer=out)
        else:
            out = numpy.ndarray((comp,nij,nkl), buffer=out)

        if cintopt is None:
            cintopt = make_cintopt(atm, bas, env, intor_name)
        prescreen = pyscf.lib.c_null_ptr()
        drv = libcgto.GTOnr2e_fill_drv
        drv(getattr(libcgto, intor_name),
            getattr(libcgto, 'GTOnr2e_fill_'+aosym), prescreen,
            out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*8)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
        cintopt = None
        return out

def getints_by_shell(intor_name, shls, atm, bas, env, comp=1):
    r'''For given 2, 3 or 4 shells, interface for libcint to get 1e, 2e,
    2-center-2e or 3-center-2e integrals

    Args:
        intor_name : str

            ==========================  =========  =============
            Function                    type       Expression
            ==========================  =========  =============
            "cint1e_ovlp_sph"           spherical  ( \| \)
            "cint1e_nuc_sph"            spherical  ( \| nuc \| \)
            "cint1e_kin_sph"            spherical  (.5 \| p dot p\)
            "cint1e_ia01p_sph"          spherical  (#C(0 1) \| nabla-rinv \| cross p\)
            "cint1e_giao_irjxp_sph"     spherical  (#C(0 1) \| r cross p\)
            "cint1e_cg_irxp_sph"        spherical  (#C(0 1) \| rc cross p\)
            "cint1e_giao_a11part_sph"   spherical  (-.5 \| nabla-rinv \| r\)
            "cint1e_cg_a11part_sph"     spherical  (-.5 \| nabla-rinv \| rc\)
            "cint1e_a01gp_sph"          spherical  (g \| nabla-rinv cross p \|\)
            "cint1e_igkin_sph"          spherical  (#C(0 .5) g \| p dot p\)
            "cint1e_igovlp_sph"         spherical  (#C(0 1) g \|\)
            "cint1e_ignuc_sph"          spherical  (#C(0 1) g \| nuc \|\)
            "cint1e_z_sph"              spherical  ( \| zc \| \)
            "cint1e_zz_sph"             spherical  ( \| zc zc \| \)
            "cint1e_r_sph"              spherical  ( \| rc \| \)
            "cint1e_r2_sph"             spherical  ( \| rc dot rc \| \)
            "cint1e_rr_sph"             spherical  ( \| rc rc \| \)
            "cint1e_pnucp_sph"          spherical  (p* \| nuc dot p \| \)
            "cint1e_prinvxp_sph"        spherical  (p* \| rinv cross p \| \)
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
            "cint1e_ipovlp_sph"         spherical  (nabla \|\)
            "cint1e_ipkin_sph"          spherical  (.5 nabla \| p dot p\)
            "cint1e_ipnuc_sph"          spherical  (nabla \| nuc \|\)
            "cint1e_iprinv_sph"         spherical  (nabla \| rinv \|\)
            "cint1e_rinv_sph"           spherical  (\| rinv \|\)
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
            "cint2e_p1vxp1_sph"         spherical  ( p* \, cross p \| \, \) ; SSO
            "cint2e_sph"                spherical  ( \, \| \, \)
            "cint2e_ig1_sph"            spherical  (#C(0 1) g \, \| \, \)
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
            "cint2e_ip1_sph"            spherical  (nabla \, \| \,\)
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
            "cint2e_ipip1_sph"          spherical  ( nabla nabla \, \| \, \)
            "cint2e_ipvip1_sph"         spherical  ( nabla \, nabla \| \, \)
            "cint2e_ip1ip2_sph"         spherical  ( nabla \, \| nabla \, \)
            "cint3c2e_ip1_sph"          spherical  (nabla \, \| \)
            "cint3c2e_ip2_sph"          spherical  ( \, \| nabla\)
            "cint2c2e_ip1_sph"          spherical  (nabla \| r12 \| \)
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
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    natm = ctypes.c_int(atm.shape[0])
    nbas = ctypes.c_int(bas.shape[0])
    if '_cart' in intor_name:
        dtype = numpy.double
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            return (l+1)*(l+2)//2 * bas[basid,NCTR_OF]
    elif '_sph' in intor_name:
        dtype = numpy.double
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            return (l*2+1) * bas[basid,NCTR_OF]
    else:
        from pyscf.gto import mole
        dtype = numpy.complex
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            k = bas[basid,KAPPA_OF]
            return mole.len_spinor(l,k) * bas[basid,NCTR_OF]
    if '3c' in intor_name:
        assert(len(shls) == 3)
        #di, dj, dk = [num_cgto_of(x) for x in shls]
        di = num_cgto_of(shls[0])
        dj = num_cgto_of(shls[1])
        l = bas[shls[2],ANG_OF]
        if '_ssc' in intor_name: # mixed spherical-cartesian
            dk = (l+1)*(l+2)//2 * bas[shls[2],NCTR_OF]
        else:
            dk = (l*2+1) * bas[shls[2],NCTR_OF]
        buf = numpy.empty((di,dj,dk,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*3)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), pyscf.lib.c_null_ptr())
        if comp == 1:
            return buf.reshape(di,dj,dk)
        else:
            return buf.transpose(3,0,1,2)
    elif '2c' in intor_name:
        assert(len(shls) == 2)
        #di, dj = [num_cgto_of(x) for x in shls]
        #buf = numpy.empty((di,dj,comp), dtype, order='F')
        di = num_cgto_of(shls[0])
        dj = num_cgto_of(shls[1])
        buf = numpy.empty((di,dj,comp), order='F') # no complex?
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), pyscf.lib.c_null_ptr())
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)
    elif '2e' in intor_name:
        assert(len(shls) == 4)
        di, dj, dk, dl = [num_cgto_of(x) for x in shls]
        buf = numpy.empty((di,dj,dk,dl,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*4)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), pyscf.lib.c_null_ptr())
        if comp == 1:
            return buf.reshape(di,dj,dk,dl)
        else:
            return buf.transpose(4,0,1,2,3)
    elif '1e' in intor_name or 'ECP' in intor_name:
        assert(len(shls) == 2)
        di, dj = [num_cgto_of(x) for x in shls]
        buf = numpy.empty((di,dj,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p))
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)
    else:
        raise RuntimeError('Unknown intor %s' % intor_name)


def make_loc(bas, key):
    if 'cart' in key:
        l = bas[:,ANG_OF]
        dims = (l+1)*(l+2)//2 * bas[:,NCTR_OF]
    elif 'sph' in key:
        dims = (bas[:,ANG_OF]*2+1) * bas[:,NCTR_OF]
    else:  # spinor
        l = bas[:,ANG_OF]
        k = bas[:,KAPPA_OF]
        dims = (l*4+2) * bas[:,NCTR_OF]
        dims[k<0] = l[k<0] * 2 + 2
        dims[k>0] = l[k>0] * 2

    ao_loc = numpy.empty(len(dims)+1, dtype=numpy.int32)
    ao_loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=ao_loc[1:])
    return ao_loc

def make_cintopt(atm, bas, env, intor):
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = c_atm.shape[0]
    nbas = c_bas.shape[0]
    cintopt = pyscf.lib.c_null_ptr()
    foptinit = getattr(libcgto, intor+'_optimizer')
    foptinit(ctypes.byref(cintopt),
             c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
             c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
             c_env.ctypes.data_as(ctypes.c_void_p))
    return ctypes.cast(cintopt, _cintoptHandler)
class _cintoptHandler(ctypes.c_void_p):
    def __del__(self):
        libcgto.CINTdel_optimizer(ctypes.byref(self))

def _stand_sym_code(sym):
    if isinstance(sym, int):
        return 's%d' % sym
    elif sym[0] in 'sS':
        return sym.lower()
    else:
        return 's' + sym.lower()


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
    mol.set_rinv_origin(mol.atom_coord(0))
    for i in range(mol.nbas):
        for j in range(mol.nbas):
            print(i, j, getints_by_shell('cint1e_prinvxp_sph', (i,j),
                                         mol._atm, mol._bas, mol._env, 3))
