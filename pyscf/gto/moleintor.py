#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
from pyscf import lib

libcgto = lib.load_library('libcgto')

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

            ================================  =============
            Function                          Expression
            ================================  =============
            "int1e_ovlp_sph"                  ( \| \)
            "int1e_nuc_sph"                   ( \| nuc \| \)
            "int1e_kin_sph"                   (.5 \| p dot p\)
            "int1e_ia01p_sph"                 (#C(0 1) \| nabla-rinv \| cross p\)
            "int1e_giao_irjxp_sph"            (#C(0 1) \| r cross p\)
            "int1e_cg_irxp_sph"               (#C(0 1) \| rc cross p\)
            "int1e_giao_a11part_sph"          (-.5 \| nabla-rinv \| r\)
            "int1e_cg_a11part_sph"            (-.5 \| nabla-rinv \| rc\)
            "int1e_a01gp_sph"                 (g \| nabla-rinv cross p \|\)
            "int1e_igkin_sph"                 (#C(0 .5) g \| p dot p\)
            "int1e_igovlp_sph"                (#C(0 1) g \|\)
            "int1e_ignuc_sph"                 (#C(0 1) g \| nuc \|\)
            "int1e_z_sph"                     ( \| zc \| \)
            "int1e_zz_sph"                    ( \| zc zc \| \)
            "int1e_r_sph"                     ( \| rc \| \)
            "int1e_r2_sph"                    ( \| rc dot rc \| \)
            "int1e_rr_sph"                    ( \| rc rc \| \)
            "int1e_pnucp_sph"                 (p* \| nuc dot p \| \)
            "int1e_prinvxp_sph"               (p* \| rinv cross p \| \)
            "int1e_ovlp_spinor"               ( \| \)
            "int1e_nuc_spinor"                ( \| nuc \|\)
            "int1e_srsr_spinor"               (sigma dot r \| sigma dot r\)
            "int1e_sr_spinor"                 (sigma dot r \|\)
            "int1e_srsp_spinor"               (sigma dot r \| sigma dot p\)
            "int1e_spsp_spinor"               (sigma dot p \| sigma dot p\)
            "int1e_sp_spinor"                 (sigma dot p \|\)
            "int1e_spnucsp_spinor"            (sigma dot p \| nuc \| sigma dot p\)
            "int1e_srnucsr_spinor"            (sigma dot r \| nuc \| sigma dot r\)
            "int1e_govlp_spinor"              (g \|\)
            "int1e_gnuc_spinor"               (g \| nuc \|\)
            "int1e_cg_sa10sa01_spinor"        (.5 sigma cross rc \| sigma cross nabla-rinv \|\)
            "int1e_cg_sa10sp_spinor"          (.5 rc cross sigma \| sigma dot p\)
            "int1e_cg_sa10nucsp_spinor"       (.5 rc cross sigma \| nuc \| sigma dot p\)
            "int1e_giao_sa10sa01_spinor"      (.5 sigma cross r \| sigma cross nabla-rinv \|\)
            "int1e_giao_sa10sp_spinor"        (.5 r cross sigma \| sigma dot p\)
            "int1e_giao_sa10nucsp_spinor"     (.5 r cross sigma \| nuc \| sigma dot p\)
            "int1e_sa01sp_spinor"             (\| nabla-rinv cross sigma \| sigma dot p\)
            "int1e_spgsp_spinor"              (g sigma dot p \| sigma dot p\)
            "int1e_spgnucsp_spinor"           (g sigma dot p \| nuc \| sigma dot p\)
            "int1e_spgsa01_spinor"            (g sigma dot p \| nabla-rinv cross sigma \|\)
            "int1e_spspsp_spinor"             (sigma dot p \| sigma dot p sigma dot p\)
            "int1e_spnuc_spinor"              (sigma dot p \| nuc \|\)
            "int1e_ovlp_cart"                 ( \| \)
            "int1e_nuc_cart"                  ( \| nuc \| \)
            "int1e_kin_cart"                  (.5 \| p dot p\)
            "int1e_ia01p_cart"                (#C(0 1) \| nabla-rinv \| cross p\)
            "int1e_giao_irjxp_cart"           (#C(0 1) \| r cross p\)
            "int1e_cg_irxp_cart"              (#C(0 1) \| rc cross p\)
            "int1e_giao_a11part_cart"         (-.5 \| nabla-rinv \| r\)
            "int1e_cg_a11part_cart"           (-.5 \| nabla-rinv \| rc\)
            "int1e_a01gp_cart"                (g \| nabla-rinv cross p \|\)
            "int1e_igkin_cart"                (#C(0 .5) g \| p dot p\)
            "int1e_igovlp_cart"               (#C(0 1) g \|\)
            "int1e_ignuc_cart"                (#C(0 1) g \| nuc \|\)
            "int1e_ipovlp_sph"                (nabla \|\)
            "int1e_ipkin_sph"                 (.5 nabla \| p dot p\)
            "int1e_ipnuc_sph"                 (nabla \| nuc \|\)
            "int1e_iprinv_sph"                (nabla \| rinv \|\)
            "int1e_rinv_sph"                  (\| rinv \|\)
            "int1e_ipovlp_spinor"             (nabla \|\)
            "int1e_ipkin_spinor"              (.5 nabla \| p dot p\)
            "int1e_ipnuc_spinor"              (nabla \| nuc \|\)
            "int1e_iprinv_spinor"             (nabla \| rinv \|\)
            "int1e_ipspnucsp_spinor"          (nabla sigma dot p \| nuc \| sigma dot p\)
            "int1e_ipsprinvsp_spinor"         (nabla sigma dot p \| rinv \| sigma dot p\)
            "int1e_ipovlp_cart"               (nabla \|\)
            "int1e_ipkin_cart"                (.5 nabla \| p dot p\)
            "int1e_ipnuc_cart"                (nabla \| nuc \|\)
            "int1e_iprinv_cart"               (nabla \| rinv \|\)
            "int1e_rinv_cart"                 (\| rinv \|\)
            "int2e_p1vxp1_sph"                ( p* \, cross p \| \, \) ; SSO
            "int2e_sph"                       ( \, \| \, \)
            "int2e_ig1_sph"                   (#C(0 1) g \, \| \, \)
            "int2e_spinor"                    (, \| \, \)
            "int2e_spsp1_spinor"              (sigma dot p \, sigma dot p \| \, \)
            "int2e_spsp1spsp2_spinor"         (sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_srsr1_spinor"              (sigma dot r \, sigma dot r \| \,\)
            "int2e_srsr1srsr2_spinor"         (sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "int2e_cg_sa10sp1_spinor"         (.5 rc cross sigma \, sigma dot p \| \,\)
            "int2e_cg_sa10sp1spsp2_spinor"    (.5 rc cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_giao_sa10sp1_spinor"       (.5 r cross sigma \, sigma dot p \| \,\)
            "int2e_giao_sa10sp1spsp2_spinor"  (.5 r cross sigma \, sigma dot p \| sigma dot p \, sigma dot p \)
            "int2e_g1_spinor"                 (g \, \| \,\)
            "int2e_spgsp1_spinor"             (g sigma dot p \, sigma dot p \| \,\)
            "int2e_g1spsp2_spinor"            (g \, \| sigma dot p \, sigma dot p\)
            "int2e_spgsp1spsp2_spinor"        (g sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_spv1_spinor"               (sigma dot p \, \| \,\)
            "int2e_vsp1_spinor"               (\, sigma dot p \| \,\)
            "int2e_spsp2_spinor"              (\, \| sigma dot p \, sigma dot p\)
            "int2e_spv1spv2_spinor"           (sigma dot p \, \| sigma dot p \,\)
            "int2e_vsp1spv2_spinor"           (\, sigma dot p \| sigma dot p \,\)
            "int2e_spv1vsp2_spinor"           (sigma dot p \, \| \, sigma dot p\)
            "int2e_vsp1vsp2_spinor"           (\, sigma dot p \| \, sigma dot p\)
            "int2e_spv1spsp2_spinor"          (sigma dot p \, \| sigma dot p \, sigma dot p\)
            "int2e_vsp1spsp2_spinor"          (\, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_ig1_cart"                  (#C(0 1) g \, \| \, \)
            "int2e_ip1_sph"                   (nabla \, \| \,\)
            "int2e_ip1_spinor"                (nabla \, \| \,\)
            "int2e_ipspsp1_spinor"            (nabla sigma dot p \, sigma dot p \| \,\)
            "int2e_ip1spsp2_spinor"           (nabla \, \| sigma dot p \, sigma dot p\)
            "int2e_ipspsp1spsp2_spinor"       (nabla sigma dot p \, sigma dot p \| sigma dot p \, sigma dot p\)
            "int2e_ipsrsr1_spinor"            (nabla sigma dot r \, sigma dot r \| \,\)
            "int2e_ip1srsr2_spinor"           (nabla \, \| sigma dot r \, sigma dot r\)
            "int2e_ipsrsr1srsr2_spinor"       (nabla sigma dot r \, sigma dot r \| sigma dot r \, sigma dot r\)
            "int2e_ip1_cart"                  (nabla \, \| \,\)
            "int2e_ssp1ssp2_spinor"           ( \, sigma dot p \| gaunt \| \, sigma dot p\)
            "int2e_cg_ssa10ssp2_spinor"       (rc cross sigma \, \| gaunt \| \, sigma dot p\)
            "int2e_giao_ssa10ssp2_spinor"     (r cross sigma  \, \| gaunt \| \, sigma dot p\)
            "int2e_gssp1ssp2_spinor"          (g \, sigma dot p  \| gaunt \| \, sigma dot p\)
            "int2e_ipip1_sph"                 ( nabla nabla \, \| \, \)
            "int2e_ipvip1_sph"                ( nabla \, nabla \| \, \)
            "int2e_ip1ip2_sph"                ( nabla \, \| nabla \, \)
            "int3c2e_ip1_sph"                 (nabla \, \| \)
            "int3c2e_ip2_sph"                 ( \, \| nabla\)
            "int2c2e_ip1_sph"                 (nabla \| r12 \| \)
            "int3c2e_spinor"                  (nabla \, \| \)
            "int3c2e_spsp1_spinor"            (nabla \, \| \)
            "int3c2e_ip1_spinor"              (nabla \, \| \)
            "int3c2e_ip2_spinor"              ( \, \| nabla\)
            "int3c2e_ipspsp1_spinor"          (nabla sigma dot p \, sigma dot p \| \)
            "int3c2e_spsp1ip2_spinor"         (sigma dot p \, sigma dot p \| nabla \)
            ================================  =============

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
            Components of the integrals, e.g. int1e_ipovlp has 3 components.
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
    >>> gto.getints('int1e_ipnuc_sph', mol._atm, mol._bas, mol._env, comp=3) # <nabla i | V_nuc | j>
    [[[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.          0.        ]
      [ 0.          0.        ]]
     [[ 0.10289944  0.48176097]
      [-0.48176097 -0.10289944]]]
    '''
    intor_name = ascint3(intor_name)
    if (intor_name.startswith('int1e') or
        intor_name.startswith('ECP') or
        intor_name.startswith('int2c2e')):
        return getints2c(intor_name, atm, bas, env, shls_slice, comp,
                         hermi, ao_loc, cintopt, out)
    elif intor_name.startswith('int2e') or intor_name.startswith('int4c1e'):
        return getints4c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    elif intor_name.startswith('int3c'):
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
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    if intor_name.endswith('_cart') or intor_name.endswith('_sph'):
        mat = numpy.ndarray((naoi,naoj,comp), numpy.double, out, order='F')
        drv_name = 'GTOint2c'
    else:
        mat = numpy.ndarray((naoi,naoj,comp), numpy.complex, out, order='F')
        if '2c2e' in intor_name:
            assert(hermi != lib.HERMITIAN and
                   hermi != lib.ANTIHERMI)
        drv_name = 'GTOint2c_spinor'

    if cintopt is None:
        cintopt = make_cintopt(atm, bas, env, intor_name)
#    cintopt = lib.c_null_ptr()

    fn = getattr(libcgto, drv_name)
    fn(getattr(libcgto, intor_name), mat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(hermi),
       (ctypes.c_int*4)(*(shls_slice[:4])),
       ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
       atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
       env.ctypes.data_as(ctypes.c_void_p))

    mat = mat.transpose(2,0,1)
    if comp == 1:
        mat = mat[0]
    return mat

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

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
        if k0 > j1 and k0 > i1:
            if 'ssc' in intor_name:
                ao_loc[k0-1:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
            elif 'spinor' in intor_name:
                ao_loc[k0-1:] = ao_loc[k0] + make_loc(bas[k0:], intor_name)

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        shape = (naoi, naoj, naok, comp)
    else:
        aosym = 's2ij'
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok, comp)

    if 'spinor' in intor_name:
        mat = numpy.ndarray(shape, numpy.complex, out, order='F')
        drv = libcgto.GTOr3c_drv
        fill = getattr(libcgto, 'GTOr3c_fill_'+aosym)
    else:
        mat = numpy.ndarray(shape, numpy.double, out, order='F')
        drv = libcgto.GTOnr3c_drv
        fill = getattr(libcgto, 'GTOnr3c_fill_'+aosym)

    if cintopt is None:
        cintopt = make_cintopt(atm, bas, env, intor_name)

    drv(getattr(libcgto, intor_name), fill,
        mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
        (ctypes.c_int*6)(*(shls_slice[:6])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    mat = numpy.rollaxis(mat, -1, 0)
    if comp == 1:
        mat = mat[0]
    return mat

def getints4c(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    aosym = _stand_sym_code(aosym)

    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]

    ao_loc = make_loc(bas, intor_name)
    if cintopt is None:
        cintopt = make_cintopt(atm, bas, env, intor_name)

    if aosym == 's8':
        assert('_spinor' not in intor_name)
        assert(shls_slice is None)
        from pyscf.scf import _vhf
        nao = ao_loc[-1]
        nao_pair = nao*(nao+1)//2
        out = numpy.ndarray((nao_pair*(nao_pair+1)//2), buffer=out)
        drv = _vhf.libcvhf.GTO2e_cart_or_sph
        drv(getattr(libcgto, intor_name), cintopt,
            out.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
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

        prescreen = lib.c_null_ptr()
        drv = libcgto.GTOnr2e_fill_drv
        drv(getattr(libcgto, intor_name),
            getattr(libcgto, 'GTOnr2e_fill_'+aosym), prescreen,
            out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*8)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
        return out

def getints_by_shell(intor_name, shls, atm, bas, env, comp=1):
    r'''For given 2, 3 or 4 shells, interface for libcint to get 1e, 2e,
    2-center-2e or 3-center-2e integrals

    Args:
        intor_name : str
            See also :func:`getints` for the supported intor_name
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
            Components of the integrals, e.g. int1e_ipovlp has 3 components.

    Returns:
        ndarray of 2-dim to 5-dim, depending on the integral type (1e,
        2e, 3c-2e, 2c2e) and the value of comp

    Examples:
        The gradients of the spherical 2e integrals

    >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
    >>> gto.getints_by_shell('int2e_ip1_sph', (0,1,0,1), mol._atm, mol._bas, mol._env, comp=3)
    [[[[[-0.        ]]]]
      [[[[-0.        ]]]]
      [[[[-0.08760462]]]]]
    '''
    intor_name = ascint3(intor_name)
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(atm.shape[0])
    nbas = ctypes.c_int(bas.shape[0])
    if intor_name.endswith('_cart'):
        dtype = numpy.double
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            return (l+1)*(l+2)//2 * bas[basid,NCTR_OF]
    elif intor_name.endswith('_sph'):
        dtype = numpy.double
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            return (l*2+1) * bas[basid,NCTR_OF]
    else:
        from pyscf.gto.mole import len_spinor
        dtype = numpy.complex
        def num_cgto_of(basid):
            l = bas[basid,ANG_OF]
            k = bas[basid,KAPPA_OF]
            return len_spinor(l,k) * bas[basid,NCTR_OF]

    null = lib.c_null_ptr()
    if intor_name.startswith('int3c'):
        assert(len(shls) == 3)
        di = num_cgto_of(shls[0])
        dj = num_cgto_of(shls[1])
        l = bas[shls[2],ANG_OF]
        if intor_name.endswith('_ssc'): # mixed spherical-cartesian
            dk = (l+1)*(l+2)//2 * bas[shls[2],NCTR_OF]
        else:
            dk = (l*2+1) * bas[shls[2],NCTR_OF]
        buf = numpy.empty((di,dj,dk,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               null, (ctypes.c_int*3)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), null, null)
        if comp == 1:
            return buf.reshape(di,dj,dk)
        else:
            return buf.transpose(3,0,1,2)

    elif intor_name.startswith('int2e') or intor_name.startswith('int4c'):
        assert(len(shls) == 4)
        di, dj, dk, dl = [num_cgto_of(x) for x in shls]
        buf = numpy.empty((di,dj,dk,dl,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               null, (ctypes.c_int*4)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), null, null)
        if comp == 1:
            return buf.reshape(di,dj,dk,dl)
        else:
            return buf.transpose(4,0,1,2,3)

    elif (intor_name.startswith('int2c') or '1e' in intor_name or
          'ECP' in intor_name):
        assert(len(shls) == 2)
        di = num_cgto_of(shls[0])
        dj = num_cgto_of(shls[1])
        buf = numpy.empty((di,dj,comp), dtype, order='F')
        fintor = getattr(libcgto, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               null, (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), null, null)
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
        dims[k<0] = (l[k<0] * 2 + 2) * bas[k<0,NCTR_OF]
        dims[k>0] = (l[k>0] * 2    ) * bas[k>0,NCTR_OF]

    ao_loc = numpy.empty(len(dims)+1, dtype=numpy.int32)
    ao_loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=ao_loc[1:])
    return ao_loc

def make_cintopt(atm, bas, env, intor):
    intor = intor.replace('_sph','').replace('_cart','').replace('_spinor','')
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = c_atm.shape[0]
    nbas = c_bas.shape[0]
    cintopt = lib.c_null_ptr()
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

def ascint3(intor_name):
    '''convert cint2 function name to cint3 function name'''
    if intor_name.startswith('cint'):
        intor_name = intor_name[1:]
    if not (intor_name.endswith('_cart') or
            intor_name.endswith('_sph') or
            intor_name.endswith('_spinor')):
        intor_name = intor_name + '_spinor'
    return intor_name


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
            print(i, j, getints_by_shell('int1e_prinvxp_sph', (i,j),
                                         mol._atm, mol._bas, mol._env, 3))
