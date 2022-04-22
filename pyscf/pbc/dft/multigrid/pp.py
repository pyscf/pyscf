import ctypes
import numpy
from pyscf import __config__
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import gamma_point

PP_WITH_RHO_CORE = getattr(__config__, 'pbc_dft_multigrid_pp_with_rho_core', True)
MIN_BLKSIZE = 13**3
MAX_BLKSIZE = 51**3

libdft = lib.load_library('libdft')

def make_rho_core(cell, mesh=None, precision=None, atm_id=None):
    if mesh is None:
        mesh = cell.mesh
    fakecell, max_radius = pp_int.fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = numpy.asarray(cell.lattice_vectors(), order='C', dtype=float)
    if abs(a - numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
        raise NotImplementedError
    eval_fn = 'make_rho_lda' + lattice_type

    b = numpy.asarray(numpy.linalg.inv(a.T), order='C', dtype=float)
    mesh = numpy.asarray(mesh, order='C', dtype=numpy.int32)
    rho_core = numpy.zeros((numpy.prod(mesh),), order='C', dtype=float)
    drv = getattr(libdft, 'build_core_density', None)
    try:
        drv(getattr(libdft, eval_fn),
            rho_core.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p),
            mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.dimension),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(max_radius))
    except Exception as e:
        raise RuntimeError("Failed to compute rho_core. %s" % e)
    return rho_core


def _get_pp_with_erf(mydf, kpts=None, max_memory=2000):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from . multigrid import _get_j_pass2
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    Gv = cell.get_Gv(mesh)
    ngrids = len(Gv)
    vpplocG = numpy.zeros((ngrids,), dtype=numpy.complex128)

    mem_avail = max(max_memory, mydf.max_memory - lib.current_memory()[0])
    blksize = min(MAX_BLKSIZE, min(ngrids, max(MIN_BLKSIZE, int(mem_avail*1e6/((cell.natm*2)*16)))))

    for ig0 in range(0, ngrids, blksize):
        ig1 = min(ngrids, ig0+blksize)
        #vpplocG_batch = pseudo.get_vlocG(cell, Gv[ig0:ig1])
        vpplocG_batch = pp_int.get_gth_vlocG_part1(cell, Gv[ig0:ig1])
        SI = cell.get_SI(Gv[ig0:ig1])
        #vpplocG[ig0:ig1] += -numpy.einsum('ij,ij->j', SI, vpplocG_batch)
        vpplocG[ig0:ig1] += -lib.multiply_sum(SI, vpplocG_batch, axis=0)

    # from get_jvloc_G0 function
    #vpplocG[0] = numpy.sum(pseudo.get_alphas(cell))
    vpp = _get_j_pass2(mydf, vpplocG, kpts_lst)[0]
    vpp2 = pp_int.get_pp_loc_part2(cell, kpts_lst)
    for k, kpt in enumerate(kpts_lst):
        vpp[k] += vpp2[k]

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    SPG_lm_aoGs = []
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            SPG_lm_aoGs.append(None)
            continue
        pp = cell._pseudo[symb]
        p1 = 0
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl > 0:
                p1 = p1+nl*(l*2+1)
        SPG_lm_aoGs.append(numpy.zeros((p1, cell.nao), dtype=numpy.complex128))

    def vppnl_by_k(cell, kpt):
        mem_avail = max(max_memory, mydf.max_memory - lib.current_memory()[0])
        blksize = min(MAX_BLKSIZE, min(ngrids, max(MIN_BLKSIZE, int(mem_avail*1e6/((48+cell.nao+13+3)*16)))))
        vppnl = 0
        for ig0 in range(0, ngrids, blksize):
            ig1 = min(ngrids, ig0+blksize)
            ng = ig1 - ig0
            # buf for SPG_lmi upto l=0..3 and nl=3
            buf = numpy.empty((48,ng), dtype=numpy.complex128)
            Gk = Gv[ig0:ig1] + kpt
            G_rad = lib.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv[ig0:ig1], kpt=kpt) * (ngrids/cell.vol)
            for ia in range(cell.natm):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        fakemol._bas[0,gto.ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)

                        p0, p1 = p1, p1+nl*(l*2+1)
                        # pYlm is real, SI[ia] is complex
                        pYlm = numpy.ndarray((nl,l*2+1,ng), dtype=numpy.complex128, buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                        #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                        #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                        #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
                if p1 > 0:
                    SPG_lmi = buf[:p1]
                    SPG_lmi *= cell.get_SI(Gv[ig0:ig1], [ia,]).conj()
                    SPG_lm_aoGs[ia] += lib.zdot(SPG_lmi, aokG)
            buf = None
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p0, p1 = p1, p1+nl*(l*2+1)
                    hl = numpy.asarray(hl)
                    SPG_lm_aoG = SPG_lm_aoGs[ia][p0:p1].reshape(nl,l*2+1,-1)
                    tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngrids**2)

    fn_pp_nl = vppnl_by_k
    if ngrids > MAX_BLKSIZE:
        fn_pp_nl = pp_int.get_pp_nl

    for k, kpt in enumerate(kpts_lst):
        vppnl = fn_pp_nl(cell, kpt)
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)


def _get_pp_without_erf(mydf, kpts=None, max_memory=2000):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    vpp = pp_int.get_pp_loc_part2(cell, kpts_lst)
    vppnl = pp_int.get_pp_nl(cell, kpts_lst)

    for k, kpt in enumerate(kpts_lst):
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl[k].real
        else:
            vpp[k] += vppnl[k]
    vppnl = None

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)


def _get_vpplocG_part1(mydf):
    cell = mydf.cell
    mesh = mydf.mesh

    if not PP_WITH_RHO_CORE:
        Gv = cell.get_Gv(mesh)
        vpplocG_part1 = pseudo.pp_int.get_pp_loc_part1_gs(cell, Gv)
    else:
        # compute rho_core in real space then transform to G space
        weight = cell.vol / numpy.prod(mesh)
        rho_core = make_rho_core(cell)
        rhoG_core = weight * tools.fft(rho_core, mesh)
        rho_core = None
        coulG = tools.get_coulG(cell, mesh=mesh)
        vpplocG_part1 = rhoG_core * coulG
        rhoG_core = coulG = None
        # G = 0 contribution
        chargs = cell.atom_charges()
        rloc = []
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            rloc.append(cell._pseudo[symb][1])
        rloc = numpy.asarray(rloc)
        vpplocG_part1[0] += 2. * numpy.pi * numpy.sum(rloc * rloc * chargs)
    return vpplocG_part1


def get_pp(mydf, kpts=None, max_memory=2000):
    if not mydf.pp_with_erf:
        mydf.vpplocG_part1 = _get_vpplocG_part1(mydf)
        return _get_pp_without_erf(mydf, kpts, max_memory)
    else:
        return _get_pp_with_erf(mydf, kpts, max_memory)


def get_vpploc_part1_ip1(mydf, kpts=numpy.zeros((1,3))):
    from .multigrid_pair import _get_j_pass2_ip1
    if mydf.pp_with_erf:
        return 0

    mesh = mydf.mesh
    vG = mydf.vpplocG_part1
    vG.reshape(-1,*mesh)

    vpp_kpts = _get_j_pass2_ip1(mydf, vG, kpts, hermi=0, deriv=1)
    if gamma_point(kpts):
        vpp_kpts = vpp_kpts.real
    if len(kpts) == 1:
        vpp_kpts = vpp_kpts[0]
    return vpp_kpts


def vpploc_part1_nuc_grad_generator(mydf, kpts=numpy.zeros((1,3))):
    from .multigrid_pair import _get_j_pass2_ip1
    h1 = -get_vpploc_part1_ip1(mydf, kpts=kpts)

    nkpts = len(kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    aoslices = cell.aoslice_by_atom()
    def hcore_deriv(atm_id):
        weight = cell.vol / numpy.prod(mesh)
        rho_core = make_rho_core(cell, atm_id=[atm_id,])
        rhoG_core = weight * tools.fft(rho_core, mesh)
        coulG = tools.get_coulG(cell, mesh=mesh)
        vpplocG_part1 = rhoG_core * coulG
        # G = 0 contribution
        symb = cell.atom_symbol(atm_id)
        rloc = cell._pseudo[symb][1]
        vpplocG_part1[0] += 2 * numpy.pi * (rloc * rloc * cell.atom_charge(atm_id))
        vpplocG_part1.reshape(-1,*mesh)
        vpp_kpts = _get_j_pass2_ip1(mydf, vpplocG_part1, kpts, hermi=0, deriv=1)
        if gamma_point(kpts):
            vpp_kpts = vpp_kpts.real
        if len(kpts) == 1:
            vpp_kpts = vpp_kpts[0]

        shl0, shl1, p0, p1 = aoslices[atm_id]
        if nkpts > 1:
            for k in range(nkpts):
                vpp_kpts[k,:,p0:p1] += h1[k,:,p0:p1]
                vpp_kpts[k] += vpp_kpts[k].transpose(0,2,1)
        else:
            vpp_kpts[:,p0:p1] += h1[:,p0:p1]
            vpp_kpts += vpp_kpts.transpose(0,2,1)
        return vpp_kpts
    return hcore_deriv


def vpploc_part1_nuc_grad(mydf, dm, kpts=numpy.zeros((1,3)), atm_id=None, precision=None):
    from .multigrid_pair import _eval_rhoG
    t0 = (logger.process_clock(), logger.perf_counter())
    cell = mydf.cell
    fakecell, max_radius = pp_int.fake_cell_vloc_part1(cell, atm_id=atm_id, precision=precision)
    atm = fakecell._atm
    bas = fakecell._bas
    env = fakecell._env

    a = numpy.asarray(cell.lattice_vectors(), order='C', dtype=float)
    if abs(a - numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
        raise NotImplementedError
    eval_fn = 'eval_mat_lda' + lattice_type + '_ip1'

    b = numpy.asarray(numpy.linalg.inv(a.T), order='C', dtype=float)
    mesh = numpy.asarray(mydf.mesh, order='C', dtype=numpy.int32)
    comp = 3
    grad = numpy.zeros((len(atm),comp), order="C", dtype=float)
    drv = getattr(libdft, 'int_gauss_charge_v_rs', None)

    if mydf.rhoG is None:
        rhoG = _eval_rhoG(mydf, dm, hermi=1, kpts=kpts, deriv=0)
    else:
        rhoG = mydf.rhoG

    ngrids = numpy.prod(mesh)
    if mydf.sccs:
        weight = cell.vol / ngrids
        rho_pol = lib.multiply(weight, mydf.sccs.rho_pol)
        rho_pol_gs = tools.fft(rho_pol, mesh).reshape(-1,ngrids)
        rhoG[:,0] += rho_pol_gs

    coulG = tools.get_coulG(cell, mesh=mesh)
    #vG = numpy.einsum('ng,g->ng', rhoG[:,0], coulG).reshape(-1,ngrids)
    vG = numpy.empty_like(rhoG[:,0], dtype=numpy.result_type(rhoG[:,0], coulG))
    for i, rhoG_i in enumerate(rhoG[:,0]):
        vG[i] = lib.multiply(rhoG_i, coulG, out=vG[i])

    v_rs = numpy.asarray(tools.ifft(vG, mesh).reshape(-1,ngrids).real, order="C")
    try:
        drv(getattr(libdft, eval_fn),
            grad.ctypes.data_as(ctypes.c_void_p),
            v_rs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp),
            atm.ctypes.data_as(ctypes.c_void_p),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p),
            mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(cell.dimension),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(max_radius))
    except Exception as e:
        raise RuntimeError("Failed to computed nuclear gradients of vpploc part1. %s" % e)
    grad *= -1
    t0 = logger.timer(mydf, 'vpploc_part1_nuc_grad', *t0)
    return grad


def get_pp_nuc_grad(mydf, kpts=numpy.zeros((1,3)), atm_id=0):
    '''
    The pseudo-potential contribution to the force for a single atom.
    '''
    cell = mydf.cell

    vpploc_part1 = vpploc_part1_nuc_grad_generator(mydf, kpts)
    vpp = vpploc_part1(atm_id)

    vpploc_part2 = pp_int.vpploc_part2_nuc_grad_generator(cell, kpts)
    vpp += numpy.asarray(vpploc_part2(atm_id))

    vppnl = pp_int.vppnl_nuc_grad_generator(cell, kpts)
    vpp += vppnl(atm_id)
    return vpp
