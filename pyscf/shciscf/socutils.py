import time
from functools import reduce

import numpy
import scipy.linalg

from pyscf import gto, lib
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.shciscf import shci
from pyscf.x2c import x2c


def print1Int(h1, name):
    xyz = ["X", "Y", "Z"]
    for k in range(3):
        with open('%s.' % (name) + xyz[k], 'w') as fout:
            fout.write('%d\n' % h1[k].shape[0])
            for i in range(h1[k].shape[0]):
                for j in range(h1[k].shape[0]):
                    if (abs(h1[k, i, j]) > 1.e-8):
                        fout.write(
                            '%16.10g %4d %4d\n' % (h1[k, i, j], i + 1, j + 1))


# by default h is returned in the contracted basis
# x and r in the uncontracted basis
def get_hxr(mc, uncontract=True):
    if (uncontract):
        xmol, contr_coeff = x2c.X2C(mc.mol).get_xmol()
    else:
        xmol, contr_coeff = mc.mol, numpy.eye(mc.mo_coeff.shape[0])

    c = lib.param.LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    s = xmol.intor_symmetric('int1e_ovlp')
    w = xmol.intor_symmetric('int1e_pnucp')

    h1, x, r = _x2c1e_hxrmat(t, v, w, s, c)
    if (uncontract):
        h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))

    return h1, x, r


def _x2c1e_hxrmat(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2, n2), dtype=v.dtype)
    m = numpy.zeros((n2, n2), dtype=v.dtype)
    h[:nao, :nao] = v
    h[:nao, nao:] = t
    h[nao:, :nao] = t
    h[nao:, nao:] = w * (.25 / c**2) - t
    m[:nao, :nao] = s
    m[nao:, nao:] = t * (.5 / c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]

    b = numpy.dot(cl, cl.T.conj())
    x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))

    s1 = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5 / c**2)
    tx = reduce(numpy.dot, (t, x))
    h1 = (h[:nao, :nao] + h[:nao, nao:].dot(x) + x.T.conj().dot(h[nao:, :nao]) +
          reduce(numpy.dot, (x.T.conj(), h[nao:, nao:], x)))

    sa = _invsqrt(s)
    sb = _invsqrt(reduce(numpy.dot, (sa, s1, sa)))
    r = reduce(numpy.dot, (sa, sb, sa, s))
    h1out = reduce(numpy.dot, (r.T.conj(), h1, r))
    return h1out, x, r


def _invsqrt(a, tol=1e-14):
    e, v = numpy.linalg.eigh(a)
    idx = e > tol
    return numpy.dot(v[:, idx] / numpy.sqrt(e[idx]), v[:, idx].T.conj())


def get_hso1e(wso, x, rp):
    nb = x.shape[0]
    hso1e = numpy.zeros((3, nb, nb))
    for ic in range(3):
        hso1e[ic] = reduce(numpy.dot, (rp.T, x.T, wso[ic], x, rp))
    return hso1e


def get_wso(mol):
    nb = mol.nao_nr()
    wso = numpy.zeros((3, nb, nb))
    for iatom in range(mol.natm):
        zA = mol.atom_charge(iatom)
        xyz = mol.atom_coord(iatom)
        mol.set_rinv_orig(xyz)
        # sign due to integration by part
        wso += zA * mol.intor('cint1e_prinvxp_sph', 3)
    return wso


def get_wso_1c(mol, atomlist):
    nb = mol.nao_nr()
    wso = numpy.zeros((3, nb, nb))
    aoslice = mol.aoslice_by_atom()
    for iatom in atomlist:
        zA = mol.atom_charge(iatom)
        xyz = mol.atom_coord(iatom)
        mol.set_rinv_orig(xyz)
        ao_start = aoslice[iatom, 2]
        ao_end = aoslice[iatom, 3]
        shl_start = aoslice[iatom, 0]
        shl_end = aoslice[iatom, 1]
        wso[:, ao_start: ao_end, ao_start: ao_end]\
            = zA*mol.intor('cint1e_prinvxp_sph', 3, shls_slice=[shl_start, shl_end, shl_start, shl_end]).reshape(3, ao_end-ao_start, ao_end-ao_start)
    return wso


def get_p(dm, x, rp):
    pLL = rp.dot(dm.dot(rp.T))
    pLS = pLL.dot(x.T)
    pSS = x.dot(pLL.dot(x.T))
    return pLL, pLS, pSS


def get_fso2e_x2c_original(mol, x, rp, pLL, pLS, pSS):
    ''' Function for x2c Hamiltonian without any memory saving '''
    # although this function is not used but just keep here for reference
    nb = mol.nao_nr()
    np = nb * nb
    nq = np * np

    ddint = mol.intor('int2e_ip1ip2_sph', 9).reshape(3, 3, nq)
    fso2e = numpy.zeros((3, nb, nb))

    xyz = [0, 1, 2]
    for i_x in xyz:
        i_y = xyz[i_x - 2]
        i_z = xyz[i_x - 1]
        ddint[0, 0] = ddint[i_y, i_z] - ddint[i_z, i_y]  # x = yz - zy etc
        kint = ddint[0, 0].reshape(nb, nb, nb, nb)
        gsoLL = -2.0 * numpy.einsum('lmkn,lk->mn', kint, pSS)
        gsoLS = -1.0 * numpy.einsum('mlkn,lk->mn', kint, pLS) \
                - 1.0 * numpy.einsum('lmkn,lk->mn', kint, pLS)
        gsoSS = -2.0 * numpy.einsum('mnkl,lk', kint, pLL) \
                - 2.0 * numpy.einsum('mnlk,lk', kint, pLL) \
            + 2.0 * numpy.einsum('mlnk,lk', kint, pLL)
        fso2e[i_x] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
            + x.T.dot(gsoSS.dot(x))
        fso2e[i_x] = reduce(numpy.dot, (rp.T, fso2e[i_x], rp))
    return fso2e


def get_fso2e_x2c(mol, x, rp, pLL, pLS, pSS):
    ''' Two-electron x2c operator with memory saving strategy '''
    nb = mol.nao_nr()

    fso2e = numpy.zeros((3, nb, nb))
    gsoLL = numpy.zeros((3, nb, nb))
    gsoLS = numpy.zeros((3, nb, nb))
    gsoSS = numpy.zeros((3, nb, nb))

    from pyscf.gto import moleintor
    nbas = mol.nbas
    max_double = mol.max_memory / 8.0 * 1.0e6
    max_basis = pow(max_double / 9., 1. / 4.)
    ao_loc_orig = moleintor.make_loc(mol._bas, 'int2e_ip1_ip2_sph')
    shl_size = []
    shl_slice = [0]
    ao_loc = [0]
    print(nb, nbas)
    print(ao_loc_orig)
    if nb > max_basis:
        for i in range(0, nbas - 1):
            if (ao_loc_orig[i + 1] - ao_loc[-1] > max_basis and ao_loc_orig[i] - ao_loc[-1]):
                ao_loc.append(ao_loc_orig[i])
                shl_size.append(ao_loc[-1] - ao_loc[-2])
                shl_slice.append(i)
    if ao_loc[-1] is not ao_loc_orig[-1]:
        ao_loc.append(ao_loc_orig[-1])
        shl_size.append(ao_loc[-1] - ao_loc[-2])
        shl_slice.append(nbas)
    print(ao_loc, shl_size, shl_slice)
    nbas = len(shl_size)

    for i in range(0, nbas):
        for j in range(0, nbas):
            for k in range(0, nbas):
                for l in range(0, nbas):
                    #start = time.clock()
                    ddint = mol.intor('int2e_ip1ip2_sph', comp=9,
                                      shls_slice=[shl_slice[i], shl_slice[i+1],
                                                  shl_slice[j], shl_slice[j+1],
                                                  shl_slice[k], shl_slice[k+1],
                                                  shl_slice[l], shl_slice[l+1]]).reshape(3, 3, -1)
                    kint = numpy.zeros(
                        3 * shl_size[i] * shl_size[j] * shl_size[k] * shl_size[l]).reshape(
                            3, shl_size[i], shl_size[j], shl_size[k], shl_size[l])
                    kint[0] = (ddint[1, 2] - ddint[2, 1]).reshape(shl_size[i],
                                                                  shl_size[j], shl_size[k], shl_size[l])
                    kint[1] = (ddint[2, 0] - ddint[0, 2]).reshape(shl_size[i],
                                                                  shl_size[j], shl_size[k], shl_size[l])
                    kint[2] = (ddint[0, 1] - ddint[1, 0]).reshape(shl_size[i],
                                                                  shl_size[j], shl_size[k], shl_size[l])
                    #end = time.clock()
                    # print("Time elapsed for integral calculation:",
                    #      end - start, i, j, k, l, nbas)

                    #start = time.clock()
                    gsoLL[:, ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]] \
                        += -2.0*numpy.einsum('ilmkn,lk->imn', kint,
                                             pSS[ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]])
                    gsoLS[:, ao_loc[i]:ao_loc[i+1], ao_loc[l]:ao_loc[l+1]] \
                        += -1.0*numpy.einsum('imlkn,lk->imn', kint,
                                             pLS[ao_loc[j]:ao_loc[j+1], ao_loc[k]:ao_loc[k+1]])
                    gsoLS[:, ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]] \
                        += -1.0*numpy.einsum('ilmkn,lk->imn', kint,
                                             pLS[ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]])
                    gsoSS[:, ao_loc[i]:ao_loc[i+1], ao_loc[j]:ao_loc[j+1]] \
                        += -2.0*numpy.einsum('imnkl,lk->imn', kint,
                                             pLL[ao_loc[l]:ao_loc[l+1], ao_loc[k]:ao_loc[k+1]])\
                        - 2.0*numpy.einsum('imnlk,lk->imn', kint,
                                           pLL[ao_loc[k]:ao_loc[k+1], ao_loc[l]:ao_loc[l+1]])
                    gsoSS[:, ao_loc[i]:ao_loc[i+1], ao_loc[k]:ao_loc[k+1]] \
                        += 2.0*numpy.einsum('imlnk,lk->imn', kint,
                                            pLL[ao_loc[j]:ao_loc[j+1], ao_loc[l]:ao_loc[l+1]])
                    #print(" Time elapsed for einsum:", time.clock() - start)

    for comp in range(0, 3):
        fso2e[comp] = gsoLL[comp]
        + gsoLS[comp].dot(x)
        + x.T.dot(-gsoLS[comp].T)
        + x.T.dot(gsoSS[comp].dot(x))
        fso2e[comp] = reduce(numpy.dot, (rp.T, fso2e[comp], rp))

    return fso2e


def get_fso2e_x2c1c(mol, x, rp, pLL, pLS, pSS):
    ''' Function for x2c Hamiltonian with one-center approximation '''
    nb = mol.nao_nr()
    #np = nb * nb
    fso2e = numpy.zeros((3, nb, nb))
    gsoLL = numpy.zeros((3, nb, nb))
    gsoLS = numpy.zeros((3, nb, nb))
    gsoSS = numpy.zeros((3, nb, nb))

    from pyscf.gto import moleintor
    #nbas = mol.nbas
    #max_double = mol.max_memory / 8.0 * 1.0e6
    #max_basis = pow(max_double / 9., 1. / 4.)
    #ao_loc_orig = moleintor.make_loc(mol._bas, 'int2e_ip1_ip2_sph')
    shl_size = []
    shl_slice = [0]
    ao_loc = [0]

    ao_slice_by_atom = mol.aoslice_by_atom()
    for slice in ao_slice_by_atom:
        shl_slice.append(slice[1])
        shl_size.append(slice[3] - slice[2])
        ao_loc.append(slice[3])
    natom = len(ao_slice_by_atom)

    for iatom in range(0, natom):
        ibegin = shl_slice[iatom]
        iend = shl_slice[iatom + 1]
        #start = time.clock()
        ddint = mol.intor('int2e_ip1ip2_sph',
                          comp=9,
                          shls_slice=[ibegin, iend, ibegin, iend, ibegin, iend, ibegin, iend]).reshape(3, 3, -1)
        kint = numpy.zeros(3 * shl_size[iatom] * shl_size[iatom] * shl_size[iatom] * shl_size[iatom]).reshape(
            3, shl_size[iatom], shl_size[iatom], shl_size[iatom], shl_size[iatom])
        kint[0] = (ddint[1, 2] - ddint[2, 1]).reshape(shl_size[iatom], shl_size[iatom], shl_size[iatom],
                                                      shl_size[iatom])
        kint[1] = (ddint[2, 0] - ddint[0, 2]).reshape(shl_size[iatom], shl_size[iatom], shl_size[iatom],
                                                      shl_size[iatom])
        kint[2] = (ddint[0, 1] - ddint[1, 0]).reshape(shl_size[iatom], shl_size[iatom], shl_size[iatom],
                                                      shl_size[iatom])
        #start = time.clock()
        gsoLL[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
            += -2.0*numpy.einsum('ilmkn,lk->imn', kint,
                                 pSS[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
        gsoLS[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
            += -1.0*numpy.einsum('imlkn,lk->imn', kint,
                                 pLS[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
        gsoLS[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
            += -1.0*numpy.einsum('ilmkn,lk->imn', kint,
                                 pLS[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
        gsoSS[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
            += -2.0*numpy.einsum('imnkl,lk->imn', kint,
                                 pLL[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])\
            - 2.0*numpy.einsum('imnlk,lk->imn', kint,
                               pLL[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
        gsoSS[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
            += 2.0*numpy.einsum('imlnk,lk->imn', kint,
                                pLL[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
        #print(" Time elapsed for einsum:", time.clock() - start)
    for comp in range(0, 3):
        fso2e[comp] = gsoLL[comp] + \
            gsoLS[comp].dot(x) + x.T.dot(-gsoLS[comp].T) + \
            x.T.dot(gsoSS[comp].dot(x))
        fso2e[comp] = reduce(numpy.dot, (rp.T, fso2e[comp], rp))

    return fso2e


def get_fso2e_bp(mol, dm):
    ''' Two-electron bp operator '''
    nb = mol.nao_nr()

    hso1e = numpy.zeros(3 * nb * nb).reshape(3, nb, nb)
    from pyscf.gto import moleintor
    nbas = mol.nbas
    max_double = mol.max_memory / 8.0 * 1.0e6
    max_basis = pow(max_double / 3., 1. / 4.)
    ao_loc_orig = moleintor.make_loc(mol._bas, 'cint2e_p1vxp1_sph')
    shl_size = []
    shl_slice = [0]
    ao_loc = [0]
    if nb > max_basis:
        for i in range(0, nbas - 1):
            if (ao_loc_orig[i + 1] - ao_loc[-1] > max_basis and ao_loc_orig[i] - ao_loc[-1]):
                ao_loc.append(ao_loc_orig[i])
                shl_size.append(ao_loc[-1] - ao_loc[-2])
                shl_slice.append(i)
    if ao_loc[-1] is not ao_loc_orig[-1]:
        ao_loc.append(ao_loc_orig[-1])
        shl_size.append(ao_loc[-1] - ao_loc[-2])
        shl_slice.append(nbas)
    nbas = len(shl_size)

    for i in range(0, nbas):
        for j in range(0, nbas):
            for k in range(0, nbas):
                for l in range(0, nbas):
                    h2ao = mol.intor('cint2e_p1vxp1_sph', comp=3, aosym='s1',
                                     shls_slice=[shl_slice[i], shl_slice[i+1],
                                                 shl_slice[j], shl_slice[j+1],
                                                 shl_slice[k], shl_slice[k+1],
                                                 shl_slice[l], shl_slice[l+1]]).reshape(
                        3, shl_size[i], shl_size[j], shl_size[k], shl_size[l])
                    hso1e[:, ao_loc[i]:ao_loc[i+1], ao_loc[j]:ao_loc[j+1]] \
                        += 1.0*numpy.einsum('ijklm, lm->ijk', h2ao,
                                            dm[ao_loc[k]:ao_loc[k+1], ao_loc[l]:ao_loc[l+1]])
                    hso1e[:, ao_loc[i]:ao_loc[i+1], ao_loc[l]:ao_loc[l+1]] \
                        += -1.5*numpy.einsum('ijklm, kl->ijm', h2ao,
                                             dm[ao_loc[j]:ao_loc[j+1], ao_loc[k]:ao_loc[k+1]])
                    hso1e[:, ao_loc[k]:ao_loc[k+1], ao_loc[j]:ao_loc[j+1]] \
                        += -1.5*numpy.einsum('ijklm, mj->ilk', h2ao,
                                             dm[ao_loc[l]:ao_loc[l+1], ao_loc[i]:ao_loc[i+1]])
    return hso1e


def get_fso2e_bp1c(mol, dm, atomlist):
    ''' Two electron bp operator with one-center approximation '''
    nb = mol.nao_nr()

    hso1e = numpy.zeros(3 * nb * nb).reshape(3, nb, nb)
    from pyscf.gto import moleintor
    max_double = mol.max_memory / 8.0 * 1.0e6
    max_basis = pow(max_double / 3., 1. / 4.)
    shl_size = []
    shl_slice = [0]
    ao_loc = [0]
    ao_slice_by_atom = mol.aoslice_by_atom()
    for slice in ao_slice_by_atom:
        shl_slice.append(slice[1])
        shl_size.append(slice[3] - slice[2])
        ao_loc.append(slice[3])

    for iatom in atomlist:
        if shl_size[iatom] > max_basis:
            print("TODO: split the basis with atom to deal with very heavy element")
        else:
            ibegin = shl_slice[iatom]
            iend = shl_slice[iatom + 1]
            #start = time.clock()
            h2ao = mol.intor('cint2e_p1vxp1_sph', comp=3, aosym='s1', shls_slice=[ibegin, iend, ibegin, iend, ibegin, iend, ibegin, iend]).reshape(
                3, shl_size[iatom], shl_size[iatom], shl_size[iatom], shl_size[iatom])
            #end = time.clock()
            # print("Time elapsed for integral calculation:",
            #      end - start, iatom, nbas)
            hso1e[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
                += 1.0*numpy.einsum('ijklm, lm->ijk', h2ao,
                                    dm[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
            hso1e[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
                += -1.5*numpy.einsum('ijklm, kl->ijm', h2ao,
                                     dm[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
            hso1e[:, ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]] \
                += -1.5*numpy.einsum('ijklm, mj->ilk', h2ao,
                                     dm[ao_loc[iatom]:ao_loc[iatom+1], ao_loc[iatom]:ao_loc[iatom+1]])
    return hso1e


def writeGTensorIntegrals(mc, atomlist=None):
    mol = mc.mol
    ncore, ncas = mc.ncore, mc.ncas
    nb = mol.nao_nr()
    if atomlist is None:
        h1ao = mol.intor('cint1e_cg_irxp_sph', comp=3)
    else:
        h1ao = numpy.zeros((3, nb, nb))
        aoslice = mc.mol.aoslice_by_atom()
        for iatom in atomlist:
            for jatom in atomlist:
                iao_start = aoslice[iatom, 2]
                jao_start = aoslice[jatom, 2]
                iao_end = aoslice[iatom, 3]
                jao_end = aoslice[jatom, 3]
                ishl_start = aoslice[iatom, 0]
                jshl_start = aoslice[jatom, 0]
                ishl_end = aoslice[iatom, 1]
                jshl_end = aoslice[jatom, 1]
                h1ao[:, iao_start: iao_end, jao_start: jao_end]\
                    += mol.intor('cint1e_cg_irxp_sph', 3, shls_slice=[ishl_start, ishl_end, jshl_start, jshl_end]).reshape(3, iao_end-iao_start, jao_end-jao_start)

    h1 = numpy.einsum('xpq, pi, qj->xij', h1ao, mc.mo_coeff, mc.mo_coeff)
    print1Int(h1[:, ncore:ncore + ncas, ncore:ncore + ncas], 'GTensor')


def writeSOCIntegrals(mc,
                      ncasorbs=None,
                      rdm1=None,
                      pictureChange1e="bp",
                      pictureChange2e="bp",
                      uncontract=True,
                      atomlist=None):

    from pyscf.lib.parameters import LIGHT_SPEED
    alpha = 1.0 / LIGHT_SPEED

    # \alpha^2/4 factor before SOC Hamiltonian in the paper
    # not applicable to ECP terms
    factor = alpha**2 * 0.25

    has_ecp = mc.mol.has_ecp()

    if ("bp" in pictureChange1e and "bp" in pictureChange2e):
        uncontract = False

    if (uncontract):
        xmol, contr_coeff = x2c.X2C(mc.mol).get_xmol()
    else:
        xmol, contr_coeff = mc.mol, numpy.eye(mc.mo_coeff.shape[0])

    if (atomlist is None):
        atomlist = numpy.linspace(0, xmol.natm - 1, xmol.natm, dtype=int)

    rdm1ao = rdm1
    if (rdm1 is None):
        rdm1ao = 1. * mc.make_rdm1()
    if len(rdm1ao.shape) > 2:
        rdm1ao = (rdm1ao[0] + rdm1ao[1])

    if (uncontract):
        dm = reduce(numpy.dot, (contr_coeff, rdm1ao, contr_coeff.T))
    else:
        dm = 1. * rdm1ao
    np, nc = contr_coeff.shape[0], contr_coeff.shape[1]

    hso1e = numpy.zeros((3, np, np))

    if ("x2c" in pictureChange1e or "x2c" in pictureChange2e):
        h1e_1c, x, rp = get_hxr(mc, uncontract=uncontract)
        if (has_ecp):
            print("X2C Hamiltonian shouldn't be used with ECPs, switch to BP.")
            exit(0)

    # ECPso terms
    if (has_ecp):
        hso1e += xmol.intor('ECPso')

    # two electron terms
    if (pictureChange2e == "bp"):
        hso1e += -factor * get_fso2e_bp(xmol, dm)
    elif (pictureChange2e == "bp1c"):
        hso1e += -factor * get_fso2e_bp1c(xmol, dm, atomlist)
    elif (pictureChange2e == "x2c"):
        pLL, pLS, pSS = get_p(dm / 2.0, x, rp)
        hso1e += -factor * get_fso2e_x2c(xmol, x, rp, pLL, pLS, pSS)
    elif (pictureChange2e == "x2c1c"):
        pLL, pLS, pSS = get_p(dm / 2.0, x, rp)
        hso1e += -factor * get_fso2e_x2c1c(xmol, x, rp, pLL, pLS, pSS)
    elif (pictureChange2e == "none"):
        hso1e += 0.0
    else:
        print(pictureChange2e, "not a valid option")
        exit(0)

    # MF 1 electron term
    if (pictureChange1e == "bp"):
        hso1e += factor * get_wso(xmol)
    elif (pictureChange1e == "x2c1"):
        wso = factor * get_wso(xmol)
        hso1e += get_hso1e(wso, x, rp)
    elif (pictureChange1e == "none"):
        hso1e += 0.0
    else:
        print(pictureChange1e, "not a valid option")
        exit(0)

    h1ao = numpy.zeros((3, nc, nc))
    if (uncontract):
        for ic in range(3):
            h1ao[ic] = reduce(
                numpy.dot, (contr_coeff.T, hso1e[ic], contr_coeff))
    else:
        h1ao = 1. * hso1e

    ncore, ncas = mc.ncore, mc.ncas
    if (ncasorbs is not None):
        ncas = ncasorbs
    mo_coeff = mc.mo_coeff
    h1 = numpy.einsum('xpq,pi,qj->xij', h1ao, mo_coeff,
                      mo_coeff)[:, ncore:ncore + ncas, ncore:ncore + ncas]
    print1Int(h1, 'SOC')


def doSOC(mc, pictureChange1e="bp", pictureChange2e="bp", uncontract=False, atomlist=None):
    writeGTensorIntegrals(mc, atomlist=atomlist)
    writeSOCIntegrals(mc, pictureChange1e=pictureChange1e,
                      pictureChange2e=pictureChange2e, uncontract=uncontract, atomlist=atomlist)
    mch = shci.SHCISCF(mc.mol, mc.norb, mc.nelec)
    mch.fcisolver.DoSOC = True
    mch.fcisolver.DoRDM = False
    shci.dryrun(mch, mc.mo_coeff)
