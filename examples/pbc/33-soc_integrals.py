#!/usr/bin/env python

'''
PBC-SOC integrals
'''

from pyscf.pbc import gto

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'ccpvdz'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.build()

#
# 1-center approximation
#
def get_1c_pvxp(cell, kpts=None):
    import numpy
    atom_slices = cell.offset_nr_by_atom()
    nao = cell.nao_nr()
    mat_soc = numpy.zeros((3,nao,nao))
    for ia in range(cell.natm):
        ish0, ish1, p0, p1 = atom_slices[ia]
        shls_slice = (ish0, ish1, ish0, ish1)
        with cell.with_rinv_as_nucleus(ia):
            z = -cell.atom_charge(ia)
            # Apply Koseki effective charge on z?
            w = z * cell.intor('int1e_prinvxp', comp=3, shls_slice=shls_slice)
        mat_soc[:,p0:p1,p0:p1] = w
    return mat_soc

#
# SOC with lattice summation (G != 0)
#
def get_pbc_pvxp(cell, kpts=None):
    import numpy
    import copy
    import time
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.pbc import tools
    from pyscf.gto import mole
    from pyscf.pbc.df import ft_ao
    from pyscf.pbc.df import aft_jk
    from pyscf.pbc.df import aft
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(cell.stdout, cell.verbose)
    t1 = t0 = (time.clock(), time.time())

    mydf = aft.AFTDF(cell, kpts)
    mydf.eta = 0.2
    ke_guess = aft.estimate_ke_cutoff_for_eta(cell, mydf.eta, cell.precision)
    mydf.mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
    log.debug('mydf.mesh %s', mydf.mesh)

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    Gv, Gvbase, kws = cell.get_Gv_weights(mydf.mesh)
    charge = -cell.atom_charges() # Apply Koseki effective charge?
    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, mesh=mydf.mesh, Gv=Gv)
    coulG *= kws
    if mydf.eta == 0:
        soc_mat = numpy.zeros((nkpts,3,nao*nao), dtype=numpy.complex128)
        SI = cell.get_SI(Gv)
        vG = numpy.einsum('i,ix->x', charge, SI) * coulG
    else:
        nuccell = copy.copy(cell)
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        norm = half_sph_norm/mole.gaussian_int(2, mydf.eta)
        chg_env = [mydf.eta, norm]
        ptr_eta = cell._env.size
        ptr_norm = ptr_eta + 1
        chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
        nuccell._atm = cell._atm
        nuccell._bas = numpy.asarray(chg_bas, dtype=numpy.int32)
        nuccell._env = numpy.hstack((cell._env, chg_env))

        soc_mat = mydf._int_nuc_vloc(nuccell, kpts_lst, 'int3c2e_pvxp1_sph',
                                     aosym='s1', comp=3)
        soc_mat = numpy.asarray(soc_mat).reshape(nkpts,3,nao**2)
        t1 = log.timer_debug1('pnucp pass1: analytic int', *t1)

        aoaux = ft_ao.ft_ao(nuccell, Gv)
        vG = numpy.einsum('i,xi->x', charge, aoaux) * coulG

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s1',
                                       intor='GTO_ft_pxp_sph', comp=3):
        for k, aoao in enumerate(aoaoks):
            aoao = aoao.reshape(3,-1,nao**2)
            if aft_jk.gamma_point(kpts_lst[k]):
                soc_mat[k] += numpy.einsum('k,ckx->cx', vG[p0:p1].real, aoao.real)
                soc_mat[k] += numpy.einsum('k,ckx->cx', vG[p0:p1].imag, aoao.imag)
            else:
                soc_mat[k] += numpy.einsum('k,ckx->cx', vG[p0:p1].conj(), aoao)
    t1 = log.timer_debug1('contracting pnucp', *t1)

    soc_mat_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if aft_jk.gamma_point(kpt):
            soc_mat_kpts.append(soc_mat[k].real.reshape(3,nao,nao))
        else:
            soc_mat_kpts.append(soc_mat[k].reshape(3,nao,nao))

    if kpts is None or numpy.shape(kpts) == (3,):
        soc_mat_kpts = soc_mat_kpts[0]
    return numpy.asarray(soc_mat_kpts)

soc_pbc = get_pbc_pvxp(cell)
soc_1c = get_1c_pvxp(cell)

print('PBC and 1-center SOC difference', abs(soc_pbc - soc_1c).max())
