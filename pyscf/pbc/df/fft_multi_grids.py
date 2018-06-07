import sys
import time
import copy
import numpy
from pyscf import lib
from pyscf.gto import ATOM_OF, NPRIM_OF, PTR_EXP, PTR_COEFF
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import numint, gen_grid
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_3d
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.gto import eval_gto
from pyscf.pbc.df import fft
from pyscf.pbc.df import ft_ao
from pyscf import __config__

sys.stderr.write('WARN: multigrid is an experimental feature. It is still in '
                 'testing\nFeatures and APIs may be changed in the future.\n')

BLKSIZE = numint.BLKSIZE
EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-1)
TO_EVEN_GRIDS = getattr(__config__, 'pbc_df_fft_multi_grids_to_even', False)
RMAX_FACTOR = getattr(__config__, 'pbc_df_fft_multi_grids_rmax_factor', 0.3)
RMAX_RATIO = getattr(__config__, 'pbc_df_fft_multi_grids_rmax_ratio', 0.75)

# RHOG_HIGH_DERIV=True will compute the high order derivatives of electron
# density in real space and FT to reciprocal space.  Set RHOG_HIGH_DERIV=False
# to approximate the density derivatives in reciprocal space (without
# evaluating the high order derivatives in real space).
RHOG_HIGH_DERIV = getattr(__config__, 'pbc_df_fft_multi_grids_rhog_high_deriv', True)

WITH_J = getattr(__config__, 'pbc_df_fft_multi_grids_with_j', True)
J_IN_XC = getattr(__config__, 'pbc_df_fft_multi_grids_j_in_xc', False)

def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    low_dim_ft_type = mydf.low_dim_ft_type
    mesh = mydf.mesh
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mesh)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv, low_dim_ft_type=low_dim_ft_type)
    vneG = rhoG * coulG
    vne = _get_j_pass2(mydf, vneG, kpts_lst)[0]

    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return numpy.asarray(vne)

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    low_dim_ft_type = mydf.low_dim_ft_type
    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv, low_dim_ft_type)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    # from get_jvloc_G0 function
    vpplocG[0] = numpy.sum(pseudo.get_alphas(cell, low_dim_ft_type))
    ngrids = len(vpplocG)

    vpp = _get_j_pass2(mydf, vpplocG, kpts_lst)[0]

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

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = numpy.empty((48,ngrids), dtype=numpy.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (ngrids/cell.vol)
        vppnl = 0
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
                    pYlm = numpy.ndarray((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngrids**2)

    for k, kpt in enumerate(kpts_lst):
        vppnl = vppnl_by_k(kpt)
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)
    rhoG = rhoG[:,0]
    coulG = tools.get_coulG(cell, mesh=cell.mesh, low_dim_ft_type=mydf.low_dim_ft_type)
    ngrids = coulG.size
    vG = numpy.einsum('ng,g->ng', rhoG.reshape(-1,ngrids), coulG)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    vj_kpts = _get_j_pass2(mydf, vG, kpts_band)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), deriv=0):
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multi_grids_tasks(cell, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    assert(deriv <= 2)
    if abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9:
        def dot_bra(bra, aodm):
            rho  = numpy.einsum('pi,pi->p', bra.real, aodm.real)
            if aodm.dtype == numpy.complex:
                rho += numpy.einsum('pi,pi->p', bra.imag, aodm.imag)
            return rho

        if deriv == 0:
            xctype = 'LDA'
            rhodim = 1
            def make_rho(ao_l, ao_h, dm_lh, dm_hl):
                c0 = lib.dot(ao_l, dm_lh)
                rho = dot_bra(ao_h, c0)
                return rho * 2

        elif deriv == 1:
            xctype = 'GGA'
            rhodim = 4
            def make_rho(ao_l, ao_h, dm_lh, dm_hl):
                ngrids = ao_l[0].shape[0]
                rho = numpy.empty((4,ngrids))
                c0 = lib.dot(ao_l[0], dm_lh)
                rho[0] = dot_bra(ao_h[0], c0)
                for i in range(1, 4):
                    rho[i] = dot_bra(ao_h[i], c0)
                c0 = lib.dot(ao_h[0], dm_hl)
                for i in range(1, 4):
                    rho[i] += dot_bra(ao_l[i], c0)
                return rho * 2  # *2 for dm_lh+dm_hl.T

        elif deriv == 2:
            xctype = 'MGGA'
            rhodim = 6
            def make_rho(ao_l, ao_h, dm_lh, dm_hl):
                ngrids = ao_l[0].shape[0]
                rho = numpy.empty((6,ngrids))
                c = [lib.dot(ao_l[i], dm_lh)
                     for i in range(4)]
                rho[0] = dot_bra(ao_h[0], c[0])
                rho[5] = 0
                for i in range(1, 4):
                    rho[i]  = dot_bra(ao_h[i], c[0])
                    rho[i] += dot_bra(ao_h[0], c[i])
                    rho[5] += dot_bra(ao_h[i], c[i]) * 2
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao_h[XX] + ao_h[YY] + ao_h[ZZ]
                rho[4] = dot_bra(ao2, c[0])
                ao2 = lib.dot(ao_l[XX]+ao_l[YY]+ao_l[ZZ], dm_lh)
                rho[4] += dot_bra(ao2, ao_h[0])
                rho[4] += rho[5] * 2
                rho[5] *= .5
                return rho * 2  # *2 for dm_lh+dm_hl.T
    else:
        raise NotImplementedError('Non-hermitian density matrices')

    ni = mydf._numint
    nx, ny, nz = cell.mesh
    rhoG = numpy.zeros((nset*rhodim,nx,ny,nz), dtype=numpy.complex)
    for grids_high, grids_low in tasks:
        cell_high = grids_high.cell
        mesh = grids_high.mesh
        coords_idx = grids_high.coords_idx
        ngrids0 = numpy.prod(mesh)
        ngrids1 = grids_high.coords.shape[0]
        log.debug('mesh %s, ngrids %s/%s', mesh, ngrids1, ngrids0)

        idx_h = grids_high.ao_idx
        dms_hh = numpy.asarray(dms[:,:,idx_h[:,None],idx_h], order='C')
        if grids_low is not None:
            idx_l = grids_low.ao_idx
            dms_hl = numpy.asarray(dms[:,:,idx_h[:,None],idx_l], order='C')
            dms_lh = numpy.asarray(dms[:,:,idx_l[:,None],idx_h], order='C')

        rho = numpy.zeros((nset,rhodim,ngrids1))
        if grids_low is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_high, kpts, deriv):
                ao_h, mask = ao_h_etc[0], ao_h_etc[2]
                for k in range(nkpts):
                    for i in range(nset):
                        rho_sub = numint.eval_rho(cell_high, ao_h[k], dms_hh[i,k],
                                                  mask, xctype, hermi)
                        rho[i,:,p0:p1] += rho_sub.real
                ao_h = ao_h_etc = None
        else:
            for ao_h_etc, ao_l_etc in zip(mydf.aoR_loop(grids_high, kpts, deriv),
                                          mydf.aoR_loop(grids_low, kpts, deriv)):
                p0, p1 = ao_h_etc[1:3]
                ao_h, mask = ao_h_etc[0][0], ao_h_etc[0][2]
                ao_l = ao_l_etc[0][0]
                for k in range(nkpts):
                    for i in range(nset):
                        rho_sub = numint.eval_rho(cell_high, ao_h[k], dms_hh[i,k],
                                                  mask, xctype, hermi)
                        rho[i,:,p0:p1] += rho_sub.real
                        rho_sub = make_rho(ao_l[k], ao_h[k], dms_lh[i,k], dms_hl[i,k])
                        rho[i,:,p0:p1] += rho_sub.real
                ao_h = ao_l = ao_h_etc = ao_l_etc = None

        rho *= 1./nkpts
        rhoR = numpy.zeros((nset*rhodim,ngrids0))
        rhoR[:,coords_idx] = rho.reshape(nset*rhodim,ngrids1)
        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        rho_freq = tools.fft(rhoR, mesh) * cell.vol/ngrids0
        for i in range(nset*rhodim):
            rhoG[i,gx[:,None,None],gy[:,None],gz] += rho_freq[i].reshape(mesh)

    return rhoG.reshape(nset,rhodim,ngrids0)


def _get_j_pass2(mydf, vG, kpts=numpy.zeros((1,3))):
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = cell.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multi_grids_tasks(cell, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    if gamma_point(kpts):
        vj_kpts = numpy.zeros((nset,nkpts,nao,nao))
    else:
        vj_kpts = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)

    ni = mydf._numint
    for grids_high, grids_low in tasks:
        mesh = grids_high.mesh
        coords_idx = grids_high.coords_idx
        ngrids0 = numpy.prod(mesh)
        ngrids1 = grids_high.coords.shape[0]
        log.debug('mesh %s, ngrids %s/%s', mesh, ngrids1, ngrids0)

        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids0)

        vR = tools.ifft(sub_vG, mesh).real.reshape(nset,ngrids0)
        vR = vR[:,coords_idx]

        idx_h = grids_high.ao_idx
        if grids_low is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_high, kpts):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        vj_sub = lib.dot(ao_h[k].conj().T*vR[i,p0:p1], ao_h[k])
                        vj_kpts[i,k,idx_h[:,None],idx_h] += vj_sub
                ao_h = ao_h_etc = None
        else:
            idx_l = grids_low.ao_idx
            for ao_h_etc, ao_l_etc in zip(mydf.aoR_loop(grids_high, kpts),
                                          mydf.aoR_loop(grids_low, kpts)):
                p0, p1 = ao_h_etc[1:3]
                ao_h = ao_h_etc[0][0]
                ao_l = ao_l_etc[0][0]
                for k in range(nkpts):
                    for i in range(nset):
                        vj_sub = lib.dot(ao_h[k].conj().T*vR[i,p0:p1], ao_h[k])
                        vj_kpts[i,k,idx_h[:,None],idx_h] += vj_sub

                        vj_sub = lib.dot(ao_h[k].conj().T*vR[i,p0:p1], ao_l[k])
                        vj_kpts[i,k,idx_h[:,None],idx_l] += vj_sub
                        vj_kpts[i,k,idx_l[:,None],idx_h] += vj_sub.conj().T
                ao_h = ao_l = ao_h_etc = ao_l_etc = None
    return vj_kpts


def rks_j_xc(mydf, dm_kpts, xc_code, hermi=1, kpts=numpy.zeros((1,3)),
             kpts_band=None, with_j=WITH_J, j_in_xc=J_IN_XC):
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    dms = None

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0

        rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k], vR)
                v[k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            add_j_(v, ao_l, ao_h, idx_l, idx_h, wv[0])

    elif xctype == 'GGA':
        deriv = 1

        if RHOG_HIGH_DERIV:
            rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)
        else:
            Gv = cell.Gv
            ngrids = Gv.shape[0]
            rhoG = numpy.empty((nset,4,ngrids), dtype=numpy.complex128)
            rhoG[:,:1] = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)
            rhoG[:,1:] = numpy.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR)
                v[k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            for k in range(nkpts):
                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wv)
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wv[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                v[k,idx_l[:,None],idx_h] += v1

    else:  # MGGA
        deriv = 2
        #TODO: RHOG_HIGH_DERIV:
        rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR)
                v[k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            for k in range(nkpts):
                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wv[:4])
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wv[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][1], wv[4], out=aow)
                v1 += lib.dot(ao_l[k][1].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][2], wv[4], out=aow)
                v1 += lib.dot(ao_l[k][2].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][3], wv[4], out=aow)
                v1 += lib.dot(ao_l[k][3].conj().T, aow)
                v[k,idx_l[:,None],idx_h] += v1

    mesh = cell.mesh
    coulG = tools.get_coulG(cell, mesh=mesh, low_dim_ft_type=mydf.low_dim_ft_type)
    ngrids = coulG.size
    vG = numpy.einsum('ng,g->ng', rhoG[:,0].reshape(-1,ngrids), coulG)
    vG = vG.reshape(nset,*mesh)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh) * (1./weight)
    rhoR = rhoR.real.reshape(nset,-1,ngrids)
    wv_freq = []
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    for i in range(nset):
        exc, vxc = ni.eval_xc(xc_code, rhoR[i], 0, deriv=1)[:2]
        if xctype == 'LDA':
            wv = vxc[0].reshape(1,ngrids)
        elif xctype == 'GGA':
            wv = numpy.empty((4,ngrids))
            vrho, vsigma = vxc[:2]
            wv[0]  = vrho
            wv[1:4] = rhoR[i,1:4] * (vsigma * 2)
        else:
            vrho, vsigma, vlapl, vtau = vxc
            wv = numpy.empty((5,ngrids))
            wv[0]  = vrho
            wv[1:4] = rhoR[i,1:4] * (vsigma * 2)
            if vlapl is None:
                wv[4] = .5*vtau
            else:
                wv[4] = (.5*vtau + 2*vlapl)

        nelec[i] += rhoR[i,0].sum() * weight
        excsum[i] += (rhoR[i,0]*exc).sum() * weight
        wv_freq.append(tools.fft(wv, mesh) * weight)

    wv_freq = numpy.asarray(wv_freq).reshape(nset,-1,*mesh)
    if j_in_xc:
        wv_freq[:,0] += vG
        vR = tools.ifft(vG.reshape(-1,ngrids), mesh)
        ecoul = numpy.einsum('ng,ng->n', rhoR[:,0].real, vR.real) * .5
        log.debug('Coulomb energy %s', ecoul)
        excsum += ecoul

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    rhoR = rhoG = None

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if gamma_point(kpts_band):
        veff = numpy.zeros((nset,nkpts,nao,nao))
        vj = numpy.zeros((nset,nkpts,nao,nao))
    else:
        veff = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)
        vj = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)

    for grids_high, grids_low in mydf.tasks:
        cell_high = grids_high.cell
        mesh = grids_high.mesh
        coords_idx = grids_high.coords_idx
        ngrids0 = numpy.prod(mesh)
        ngrids1 = grids_high.coords.shape[0]
        log.debug('mesh %s, ngrids %s/%s', mesh, ngrids1, ngrids0)

        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        sub_wvG = wv_freq[:,:,gx[:,None,None],gy[:,None],gz].reshape(-1,ngrids0)
        wv = tools.ifft(sub_wvG, mesh).real.reshape(nset,-1,ngrids0)
        wv = wv[:,:,coords_idx]
        if with_j:
            sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(-1,ngrids0)
            vR = tools.ifft(sub_vG, mesh).real.reshape(nset,ngrids0)
            vR = vR[:,coords_idx]

        idx_h = grids_high.ao_idx
        if grids_low is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_high, kpts, deriv):
                ao_h = ao_h_etc[0]
                for i in range(nset):
                    add_xc_(veff[i], ao_h, ao_h, idx_h, idx_h, wv[i,:,p0:p1])
                    if with_j:
                        add_j_(vj[i], ao_h, ao_h, idx_h, idx_h, vR[i,p0:p1])
                ao_h = ao_h_etc = None
        else:
            idx_l = grids_low.ao_idx
            for ao_h_etc, ao_l_etc in zip(mydf.aoR_loop(grids_high, kpts, deriv),
                                          mydf.aoR_loop(grids_low, kpts, deriv)):
                p0, p1 = ao_h_etc[1:3]
                ao_h = ao_h_etc[0][0]
                ao_l = ao_l_etc[0][0]
                for i in range(nset):
                    add_xc_(veff[i], ao_h, ao_h, idx_h, idx_h, wv[i,:,p0:p1])
                    add_xc_(veff[i], ao_h, ao_l, idx_h, idx_l, wv[i,:,p0:p1])
                    add_xc_(veff[i], ao_l, ao_h, idx_l, idx_h, wv[i,:,p0:p1])
                    if with_j:
                        add_j_(vj[i], ao_h, ao_h, idx_h, idx_h, vR[i,p0:p1])
                        add_j_(vj[i], ao_h, ao_l, idx_h, idx_l, vR[i,p0:p1])
                        add_j_(vj[i], ao_l, ao_h, idx_l, idx_h, vR[i,p0:p1])
                ao_h = ao_l = ao_h_etc = ao_l_etc = None

    vj = _format_jks(vj, dm_kpts, input_band, kpts)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    return nelec, excsum, veff, vj


# Can handle only one set of KUKS density matrices (compare to multiple sets
# of KRKS density matrices in rks_j_xc)
def uks_j_xc(mydf, dm_kpts, xc_code, hermi=1, kpts=numpy.zeros((1,3)),
             kpts_band=None, with_j=WITH_J, j_in_xc=J_IN_XC):
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    dms = None
    #TODO: Handle multiple sets of KUKS density matrices (2,nset,nkpts,nao,nao)
    assert(nset == 2)  # alpha and beta density matrices in KUKS

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0

        rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k], vR[0])
                v[0,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k])
                aow = numpy.einsum('pi,p->pi', ao_l[k], vR[1])
                v[1,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            add_j_(v, ao_l, ao_h, idx_l, idx_h, wv[:,0])

    elif xctype == 'GGA':
        deriv = 1

        if RHOG_HIGH_DERIV:
            rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)
        else:
            Gv = cell.Gv
            ngrids = Gv.shape[0]
            rhoG = numpy.empty((2,4,ngrids), dtype=numpy.complex128)
            rhoG[:,:1] = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)
            rhoG[:,1:] = numpy.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR[0])
                v[0,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR[1])
                v[1,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            wva, wvb = wv
            for k in range(nkpts):
                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wva)
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wva[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                v[0,k,idx_l[:,None],idx_h] += v1
                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wvb)
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wvb[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                v[1,k,idx_l[:,None],idx_h] += v1

    else:  # MGGA
        deriv = 2
        #TODO: RHOG_HIGH_DERIV:
        rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

        def add_j_(v, ao_l, ao_h, idx_l, idx_h, vR):
            for k in range(nkpts):
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR[0])
                v[0,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('pi,p->pi', ao_l[k][0], vR[1])
                v[1,k,idx_l[:,None],idx_h] += lib.dot(aow.conj().T, ao_h[k][0])

        def add_xc_(v, ao_l, ao_h, idx_l, idx_h, wv):
            wva, wvb = wv
            for k in range(nkpts):
                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wva[:4])
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wva[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][1], wva[4], out=aow)
                v1 += lib.dot(ao_l[k][1].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][2], wva[4], out=aow)
                v1 += lib.dot(ao_l[k][2].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][3], wva[4], out=aow)
                v1 += lib.dot(ao_l[k][3].conj().T, aow)
                v[0,k,idx_l[:,None],idx_h] += v1

                aow = numpy.einsum('npi,np->pi', ao_l[k][:4], wvb[:4])
                v1  = lib.dot(aow.conj().T, ao_h[k][0])
                aow = numpy.einsum('npi,np->pi', ao_h[k][1:4], wvb[1:4])
                v1 += lib.dot(ao_l[k][0].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][1], wvb[4], out=aow)
                v1 += lib.dot(ao_l[k][1].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][2], wvb[4], out=aow)
                v1 += lib.dot(ao_l[k][2].conj().T, aow)
                aow = numpy.einsum('pi,p->pi', ao_h[k][3], wvb[4], out=aow)
                v1 += lib.dot(ao_l[k][3].conj().T, aow)
                v[1,k,idx_l[:,None],idx_h] += v1

    mesh = cell.mesh
    coulG = tools.get_coulG(cell, mesh=mesh, low_dim_ft_type=mydf.low_dim_ft_type)
    ngrids = coulG.size
    vG = numpy.einsum('ng,g->ng', rhoG[:,0].reshape(-1,ngrids), coulG)
    vG = vG.reshape(2,*mesh)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh) * (1./weight)
    rhoR = rhoR.real.reshape(2,-1,ngrids)
    nelec = numpy.zeros(2)
    excsum = 0

    exc, vxc = ni.eval_xc(xc_code, rhoR, 1, deriv=1)[:2]
    if xctype == 'LDA':
        vrho = vxc[0]
        wva = vrho[:,0].reshape(1,ngrids)
        wvb = vrho[:,1].reshape(1,ngrids)
    elif xctype == 'GGA':
        vrho, vsigma = vxc[:2]
        wva = numpy.empty((4,ngrids))
        wvb = numpy.empty((4,ngrids))
        wva[0]  = vrho[:,0]
        wva[1:4] = rhoR[0,1:4] * (vsigma[:,0] * 2)  # sigma_uu
        wva[1:4]+= rhoR[1,1:4] *  vsigma[:,1]       # sigma_ud
        wvb[0]  = vrho[:,1]
        wvb[1:4] = rhoR[1,1:4] * (vsigma[:,2] * 2)  # sigma_dd
        wvb[1:4]+= rhoR[0,1:4] *  vsigma[:,1]       # sigma_ud
    else:
        vrho, vsigma, vlapl, vtau = vxc
        wva = numpy.empty((5,ngrids))
        wvb = numpy.empty((5,ngrids))
        wva[0]  = vrho[:,0]
        wva[1:4] = rhoR[0,1:4] * (vsigma[:,0] * 2)  # sigma_uu
        wva[1:4]+= rhoR[1,1:4] *  vsigma[:,1]       # sigma_ud
        wvb[0]  = vrho[:,1]
        wvb[1:4] = rhoR[1,1:4] * (vsigma[:,2] * 2)  # sigma_dd
        wvb[1:4]+= rhoR[0,1:4] *  vsigma[:,1]       # sigma_ud
        if vlapl is None:
            wvb[4] = .5*vtau[:,1]
            wva[4] = .5*vtau[:,0]
        else:
            wva[4] = (.5*vtau[:,0] + 2*vlapl[:,0])
            wvb[4] = (.5*vtau[:,1] + 2*vlapl[:,1])

    nelec[0] += rhoR[0,0].sum() * weight
    nelec[1] += rhoR[1,0].sum() * weight
    excsum += (rhoR[0,0]*exc).sum() * weight
    excsum += (rhoR[1,0]*exc).sum() * weight
    wv_freq = tools.fft(numpy.vstack((wva,wvb)), mesh) * weight
    wv_freq = wv_freq.reshape(2,-1,*mesh)
    if j_in_xc:
        wv_freq[:,0] += vG
        vR = tools.ifft(vG.reshape(-1,ngrids), mesh)
        ecoul = numpy.einsum('ng,ng->', rhoR[:,0].real, vR.real) * .5
        log.debug('Coulomb energy %s', ecoul)
        excsum += ecoul
    rhoR = rhoG = None

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if gamma_point(kpts_band):
        veff = numpy.zeros((2,nkpts,nao,nao))
        vj = numpy.zeros((2,nkpts,nao,nao))
    else:
        veff = numpy.zeros((2,nkpts,nao,nao), dtype=numpy.complex128)
        vj = numpy.zeros((2,nkpts,nao,nao), dtype=numpy.complex128)

    for grids_high, grids_low in mydf.tasks:
        cell_high = grids_high.cell
        mesh = grids_high.mesh
        coords_idx = grids_high.coords_idx
        ngrids0 = numpy.prod(mesh)
        ngrids1 = grids_high.coords.shape[0]
        log.debug('mesh %s, ngrids %s/%s', mesh, ngrids1, ngrids0)

        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        sub_wvG = wv_freq[:,:,gx[:,None,None],gy[:,None],gz].reshape(-1,ngrids0)
        wv = tools.ifft(sub_wvG, mesh).real.reshape(2,-1,ngrids0)
        wv = wv[:,:,coords_idx]
        if with_j:
            sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(-1,ngrids0)
            vR = tools.ifft(sub_vG, mesh).real.reshape(2,ngrids0)
            vR = vR[:,coords_idx]

        idx_h = grids_high.ao_idx
        if grids_low is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_high, kpts, deriv):
                ao_h = ao_h_etc[0]
                add_xc_(veff, ao_h, ao_h, idx_h, idx_h, wv[:,:,p0:p1])
                if with_j:
                    add_j_(vj, ao_h, ao_h, idx_h, idx_h, vR[:,p0:p1])
                ao_h = ao_h_etc = None
        else:
            idx_l = grids_low.ao_idx
            for ao_h_etc, ao_l_etc in zip(mydf.aoR_loop(grids_high, kpts, deriv),
                                          mydf.aoR_loop(grids_low, kpts, deriv)):
                p0, p1 = ao_h_etc[1:3]
                ao_h = ao_h_etc[0][0]
                ao_l = ao_l_etc[0][0]
                add_xc_(veff, ao_h, ao_h, idx_h, idx_h, wv[:,:,p0:p1])
                add_xc_(veff, ao_h, ao_l, idx_h, idx_l, wv[:,:,p0:p1])
                add_xc_(veff, ao_l, ao_h, idx_l, idx_h, wv[:,:,p0:p1])
                if with_j:
                    add_j_(vj, ao_h, ao_h, idx_h, idx_h, vR[:,p0:p1])
                    add_j_(vj, ao_h, ao_l, idx_h, idx_l, vR[:,p0:p1])
                    add_j_(vj, ao_l, ao_h, idx_l, idx_h, vR[:,p0:p1])
                ao_h = ao_l = ao_h_etc = ao_l_etc = None

    vj = _format_jks(vj, dm_kpts, input_band, kpts)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    return nelec, excsum, veff, vj


def multi_grids_tasks(cell, verbose=None):
    log = lib.logger.new_logger(cell, verbose)
    tasks = []

    a = cell.lattice_vectors()
    neighbour_images = lib.cartesian_prod(([0, -1, 1],
                                           [0, -1, 1],
                                           [0, -1, 1]))
    # Remove the first one which is the unit cell itself
    neighbour_images0 = neighbour_images
    neighbour_images = neighbour_images[1:]
    neighbour_images = neighbour_images.dot(a)
    b = numpy.linalg.inv(a.T)
    heights = 1. / numpy.linalg.norm(b, axis=1)
    normal_vector = b * heights.reshape(-1,1)
    distance_to_edge = cell.atom_coords().dot(normal_vector.T)
    #FIXME: if atoms out of unit cell
    #assert(numpy.all(distance_to_edge >= 0))
    distance_to_edge = numpy.hstack([distance_to_edge, heights-distance_to_edge])
    min_distance_to_edge = distance_to_edge.min(axis=1)

    # Split shells based on rcut
    rcuts_pgto, kecuts_pgto = _primitive_gto_cutoff(cell)
    ao_loc = cell.ao_loc_nr()

    def make_cell_high_exp(shls_high, r0, r1):
        cell_high = copy.copy(cell)
        cell_high._bas = cell._bas.copy()
        cell_high._env = cell._env.copy()

        rcut_atom = [0] * cell.natm
        ke_cutoff = 0
        for ib in shls_high:
            rc = rcuts_pgto[ib]
            idx = numpy.where((r1 <= rc) & (rc < r0))[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_high._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_high._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_high._bas[ib,NPRIM_OF] = np1

            ke_cutoff = max(ke_cutoff, kecuts_pgto[ib][idx].max())

            ia = cell.bas_atom(ib)
            rcut_atom[ia] = max(rcut_atom[ia], rc[idx].max())
        cell_high._bas = cell_high._bas[shls_high]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_high])
        return cell_high, ao_idx, ke_cutoff, rcut_atom

    def make_cell_low_exp(shls_low, r0, r1):
        cell_low = copy.copy(cell)
        cell_low._bas = cell._bas.copy()
        cell_low._env = cell._env.copy()

        for ib in shls_low:
            idx = numpy.where(r0 <= rcuts_pgto[ib])[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_low._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_low._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_low._bas[ib,NPRIM_OF] = np1
        cell_low._bas = cell_low._bas[shls_low]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_low])
        return cell_low, ao_idx

    nao = cell.nao_nr()
    rmax = a.max() * RMAX_FACTOR
    n_delimeter = int(numpy.log(0.01/rmax) / numpy.log(RMAX_RATIO))
    rcut_delimeter = rmax * (RMAX_RATIO ** numpy.arange(n_delimeter))
    for r0, r1 in zip(numpy.append(1e9, rcut_delimeter),
                      numpy.append(rcut_delimeter, 0)):
        # shells which have high exps (small rcut)
        shls_high = [ib for ib, rc in enumerate(rcuts_pgto)
                     if numpy.any((r1 <= rc) & (rc < r0))]
        if len(shls_high) == 0:
            continue
        cell_high, ao_idx_high, ke_cutoff, rcut_atom = \
                make_cell_high_exp(shls_high, r0, r1)

        # shells which have low exps (big rcut)
        shls_low = [ib for ib, rc in enumerate(rcuts_pgto)
                     if numpy.any(r0 <= rc)]
        if len(shls_low) == 0:
            cell_low = None
            ao_idx_low = []
        else:
            cell_low, ao_idx_low = make_cell_low_exp(shls_low, r0, r1)

        mesh = tools.cutoff_to_mesh(a, ke_cutoff)
        if TO_EVEN_GRIDS:
            mesh = (mesh+1)//2 * 2  # to the nearest even number
        if numpy.all(mesh >= cell.mesh):
            # Including all rest shells
            shls_high = [ib for ib, rc in enumerate(rcuts_pgto)
                         if numpy.any(rc < r0)]
            cell_high, ao_idx_high = make_cell_high_exp(shls_high, r0, 0)[:2]
        cell_high.mesh = mesh = numpy.min([mesh, cell.mesh], axis=0)

        coords = cell.gen_uniform_grids(mesh)
        coords_f4 = coords.astype(numpy.float32)
        ngrids = coords_f4.shape[0]
        coords_idx = numpy.zeros(ngrids, dtype=bool)
        gg = numpy.einsum('px,px->p', coords_f4, coords_f4)
        Lg = (2*neighbour_images.astype(numpy.float32)).dot(coords_f4.T)
        for ia in set(cell_high._bas[:,ATOM_OF]):
            rcut = rcut_atom[ia]
            log.debug1('        atom %d rcut %g', mesh, ia, rcut)

            atom_coord = cell.atom_coord(ia)
            #dr = coords_f4 - atom_coord.astype(numpy.float32)
            #coords_idx |= numpy.einsum('px,px->p', dr, dr) <= rcut**2
            #optimized to
            gg_ag = gg - coords_f4.dot(2*atom_coord.astype(numpy.float32))
            coords_idx |= gg_ag <= rcut**2 - atom_coord.dot(atom_coord)

            if min_distance_to_edge[ia] > rcut:
                # atom + rcut is completely inside the unit cell
                continue

            atoms_in_neighbour = neighbour_images + atom_coord
            distance_to_unit_cell = atoms_in_neighbour.dot(normal_vector.T)
            distance_to_unit_cell = numpy.hstack([abs(distance_to_unit_cell),
                                                  abs(heights-distance_to_unit_cell)])
            idx = distance_to_unit_cell.min(axis=1) <= rcut
            #for r_atom in atoms_in_neighbour[idx]:
            #    dr = coords_f4 - r_atom.astype(numpy.float32)
            #    coords_idx |= numpy.einsum('px,px->p', dr, dr) <= rcut**2
            #optimized to
            for i in numpy.where(idx)[0]:
                L_a = atoms_in_neighbour[i]
                coords_idx |= gg_ag - Lg[i] <= rcut**2 - L_a.dot(L_a)

        coords = coords[coords_idx]
        grids_high = gen_grid.UniformGrids(cell_high)
        grids_high.coords = coords
        grids_high.non0tab = grids_high.make_mask(cell_high, coords)
        grids_high.coords_idx = coords_idx
        grids_high.ao_idx = ao_idx_high

        if cell_low is None:
            grids_low = None
        else:
            grids_low = gen_grid.UniformGrids(cell_low)
            grids_low.coords = coords
            grids_low.non0tab = grids_low.make_mask(cell_low, coords)
            grids_low.coords_idx = coords_idx
            grids_low.ao_idx = ao_idx_low

        log.debug('mesh %s nao all/high/low %d %d %d, ngrids %d',
                  mesh, nao, len(ao_idx_high), len(ao_idx_low), coords.shape[0])

        tasks.append([grids_high, grids_low])
        if numpy.all(mesh >= cell.mesh):
            break
    return tasks

def _primitive_gto_cutoff(cell):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    precision = cell.precision * EXTRA_PREC
    log_prec = numpy.log(precision)
    b = cell.reciprocal_vectors(norm_to=1)
    ke_factor = abs(numpy.linalg.det(b))
    rcut = []
    ke_cutoff = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5

        ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, precision, ke_factor)

        rcut.append(r)
        ke_cutoff.append(ke_guess)
    return rcut, ke_cutoff


class MultiGridFFTDF(fft.FFTDF):
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        fft.FFTDF.__init__(self, cell, kpts)
        self.tasks = None
        self._keys = self._keys.union(['tasks'])

    def build(self):
        self.tasks = multi_grids_tasks(self.cell)

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        assert(not with_k)

        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            #vj, vk = fft_jk.get_jk(self, dm, hermi, kpts, kpts_band,
            #                       with_j, with_k, exxdiv)
            vj = get_j_kpts(self, dm, hermi, kpts.reshape(1,3), kpts_band)
            if kpts_band is None:
                vj = vj[...,0,:,:]
        else:
            #if with_k:
            #    vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_j_kpts = get_j_kpts
    rks_j_xc = rks_j_xc
    uks_j_xc = uks_j_xc


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft
    from pyscf.pbc import df
    from pyscf.pbc.df import fft_jk
    cell = gto.M(
        a = numpy.eye(3)*3.5668,
        atom = '''C     0.      0.      0.    
                  C     0.8917  0.8917  0.8917
                  #C     1.7834  1.7834  0.    
                  #C     2.6751  2.6751  0.8917
                  #C     1.7834  0.      1.7834
                  #C     2.6751  0.8917  2.6751
                  #C     0.      1.7834  1.7834
                  #C     0.8917  2.6751  2.6751''',
        #basis = 'sto3g',
        #basis = 'ccpvdz',
        #basis = 'gth-dzvp',
        #basis = 'unc-gth-szv',
        basis = 'gth-szv',
        #basis = [#[0, (3,1)],
        #         [0, (0.2, 1)]],
        #verbose = 5,
        #mesh = [15]*3,
        #precision=1e-6
    )

    mydf = df.FFTDF(cell)
    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = cell.make_kpts([3,1,1])
    dm = numpy.random.random((len(kpts),nao,nao)) * .2
    dm += numpy.eye(nao)
    dm = dm + dm.transpose(0,2,1)
    t0 = time.time()
    print(time.clock())
    ref = 0
    ref = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock(), time.time()-t0)
    mydf = MultiGridFFTDF(cell)
    v = get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock(), time.time()-t0)
    print('diff', abs(ref-v).max(), lib.finger(v)-lib.finger(ref))

    print(time.clock())
    mydf = df.FFTDF(cell)
    mydf.grids.build()
    n, exc, ref = mydf._numint.nr_rks(cell, mydf.grids, 'tpss', dm, 0, kpts)
    print(time.clock())
    RMAX_FACTOR = .5
    RHOG_HIGH_DERIV = False
    mydf = MultiGridFFTDF(cell)
    n, exc, vxc, vj = rks_j_xc(mydf, dm, 'tpss', kpts=kpts, with_j=False)
    print(time.clock())
    print('diff', abs(ref-vxc).max(), lib.finger(vxc)-lib.finger(ref))

    cell1 = gto.Cell()
    cell1.verbose = 0
    cell1.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell1.a = numpy.diag([4, 4, 4])
    cell1.basis = 'gth-szv'
    cell1.pseudo = 'gth-pade'
    cell1.mesh = [20]*3
    cell1.build()
    k = numpy.ones(3)*.25
    mydf = MultiGridFFTDF(cell1)
    v1 = get_pp(mydf, k)
    print(lib.finger(v1) - (1.8428463642697195-0.10478381725330854j))
    v1 = get_nuc(mydf, k)
    print(lib.finger(v1) - (2.3454744614944714-0.12528407127454744j))

    kpts = cell.make_kpts([2,2,2])
    mf = dft.KRKS(cell, kpts)
    mf.verbose = 4
    mf.with_df = MultiGridFFTDF(cell, kpts)
    mf.xc = xc = 'lda,vwn'
    def get_veff(cell, dm, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        if kpts is None:
            kpts = mf.with_df.kpts
        n, exc, vxc, vj = mf.with_df.rks_j_xc(dm, mf.xc, kpts=kpts, kpts_band=kpts_band)
        weight = 1./len(kpts)
        ecoul = numpy.einsum('Kij,Kji', dm, vj).real * .5 * weight
        vxc += vj
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
        return vxc
    mf.get_veff = get_veff
    mf.kernel()
