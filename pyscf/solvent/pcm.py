# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import scipy
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent

libdft = lib.load_library('libdft')

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


# Inject ddPCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf

# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
6: 4.84566077868,
14: 4.86458714334,
26: 4.85478226219,
38: 4.90105812685,
50: 4.89250673295,
86: 4.89741372580,
110: 4.90101060987,
146: 4.89825187392,
170: 4.90685517725,
194: 4.90337644248,
302: 4.90498088169,
350: 4.86879474832,
434: 4.90567349080,
590: 4.90624071359,
770: 4.90656435779,
974: 4.90685167998,
1202: 4.90704098216,
1454: 4.90721023869,
1730: 4.90733270691,
2030: 4.90744499142,
2354: 4.90753082825,
2702: 4.90760972766,
3074: 4.90767282394,
3470: 4.90773141371,
3890: 4.90777965981,
4334: 4.90782469526,
4802: 4.90749125553,
5294: 4.90762073452,
5810: 4.90792902522,
}

Bondi = radii.VDW
Bondi[1] = 1.1/radii.BOHR      # modified version
#radii_table = bondi * 1.2
PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)  
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4 
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def gen_surface(mol, ng=302, vdw_scale=1.2):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = numpy.empty((ng,4))
    libdft.MakeAngularGrid(unit_sphere.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ng))

    atom_coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()
    N_J = ng * numpy.ones(mol.natm)
    R_J = numpy.asarray([vdw_scale*Bondi[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J
    
    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg = gto.charge(symb)
        r_vdw = vdw_scale*Bondi[chg]
        
        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        fiJ = switch_h(diJ)
        
        w = unit_sphere[:,3] * 4.0 * PI
        swf = numpy.prod(fiJ, axis=1) 
        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        xi = XI[ng] / (r_vdw * w[idx]**0.5)
        charge_exp.append(xi)
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])
    
    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)
    
    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords
    }
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A

def get_dF_dA(surface):
    '''
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    '''

    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    R_in_J      = surface['R_in_J']
    R_sw_J      = surface['R_sw_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    dF = numpy.zeros([ngrids, natom, 3])
    dA = numpy.zeros([ngrids, natom, 3])
    
    for ia in range(atom_coords.shape[0]):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        p1 = p0 + coords.shape[0]
        ri_rJ = numpy.expand_dims(coords, axis=1) - atom_coords
        riJ = numpy.linalg.norm(ri_rJ, axis=-1)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        ri_rJ[:,ia,:] = 0.0
        ri_rJ[diJ < 1e-8] = 0.0

        fiJ = switch_h(diJ)
        dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
        dfiJ = numpy.expand_dims(dfiJ, axis=-1) * ri_rJ

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]
        
        # grids response
        Fi = numpy.expand_dims(Fi, axis=-1)
        Ai = numpy.expand_dims(Ai, axis=-1)
        dFi_grid = numpy.sum(dfiJ, axis=1)
        
        dF[p0:p1,ia,:] += Fi * dFi_grid
        dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = numpy.expand_dims(Fi, axis=-2)
        Ai = numpy.expand_dims(Ai, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        dA[p0:p1,:,:] -= Ai * dfiJ
    
    return dF, dA

def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    S = scipy.special.erf(xi_r_ij) / rij
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)
    
    D = None
    if with_D:
        drij = numpy.expand_dims(grid_coords, axis=1) - grid_coords
        nrij = numpy.sum(drij * norm_vec, axis=-1)
        
        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S

def get_dD_dS(surface, dF, with_S=True, with_D=False):
    '''
    derivative of D and S w.r.t grids, partial_i D_ij = -partial_j D_ij
    S is symmetric, D is not
    '''
    grid_coords = surface['grid_coords']
    exponents   = surface['charge_exp']
    norm_vec    = surface['norm_vec']
    switch_fun  = surface['switch_fun']

    xi_i, xi_j = numpy.meshgrid(exponents, exponents, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    ri_rj = numpy.expand_dims(grid_coords, axis=1) - grid_coords
    rij = numpy.linalg.norm(ri_rj, axis=-1)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    
    dS_dr = -(scipy.special.erf(xi_r_ij) - 2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2))/rij**2
    numpy.fill_diagonal(dS_dr, 0)
    
    dS_dr= numpy.expand_dims(dS_dr, axis=-1)
    drij = ri_rj/numpy.expand_dims(rij, axis=-1)
    dS = dS_dr * drij

    dD = None
    if with_D:
        nj_rij = numpy.sum(ri_rj * norm_vec, axis=-1)
        dD_dri = 4.0*xi_r_ij**2 * xi_ij / PI**0.5 * numpy.exp(-xi_r_ij**2) * nj_rij / rij**3
        numpy.fill_diagonal(dD_dri, 0.0)
        
        rij = numpy.expand_dims(rij, axis=-1)
        nj_rij = numpy.expand_dims(nj_rij, axis=-1)
        nj = numpy.expand_dims(norm_vec, axis=0)
        dD_dri = numpy.expand_dims(dD_dri, axis=-1)
        
        dD = dD_dri * drij + dS_dr * (-nj/rij + 3.0*nj_rij/rij**2 * drij)

    dSii_dF = -exponents * (2.0/PI)**0.5 / switch_fun**2
    dSii = numpy.expand_dims(dSii_dF, axis=(1,2)) * dF

    return dD, dS, dSii

def grad_kernel(pcmobj, dm):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR - dK q)
    '''
    mol = pcmobj.mol
    nao = mol.nao
    aoslice = mol.aoslice_by_atom()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']
    switch_fun   = pcmobj.surface['switch_fun']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    R            = pcmobj._intermediates['R']
    K            = pcmobj._intermediates['K']
    q            = pcmobj._intermediates['q']
    q_sym        = pcmobj._intermediates['q_sym']

    vK_1 = numpy.linalg.solve(K.T, v_grids)

    # ----------------- potential response -----------------------
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))
    ngrids = grid_coords.shape[0]
    atom_coords = mol.atom_coords(unit='B')

    dvj = numpy.zeros([nao,3])
    dq = numpy.zeros([ngrids,3])
    for p0, p1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents**2)
        # charge response
        v_nj_ip1 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1', aosym='s1', comp=3)
        vj = numpy.einsum('xijn,n->xij', v_nj_ip1, q_sym)
        dvj += numpy.einsum('xij,ij->ix', vj, dm)
        dvj += numpy.einsum('xij,ji->ix', vj, dm)

        # electronic potential response
        v_nj_ip2 = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip2', aosym='s1', comp=3)
        dq_slice = numpy.einsum('xijn,ij->nx', v_nj_ip2, dm)
        dq[p0:p1] = numpy.einsum('nx,n->nx', dq_slice, q_sym[p0:p1])

    de = numpy.zeros_like(atom_coords)        
    de += numpy.asarray([numpy.sum(dq[p0:p1], axis=0) for p0,p1 in gridslice])
    de += numpy.asarray([numpy.sum(dvj[p0:p1], axis=0) for p0,p1 in aoslice[:,2:]])
    
    atom_charges = mol.atom_charges()
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    
    # nuclei response
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    dv_g = numpy.einsum('g,xng->nx', q_sym, v_ng_ip1)
    de -= numpy.einsum('nx,n->nx', dv_g, atom_charges)

    # nuclei potential response
    int2c2e_ip2 = mol._add_suffix('int2c2e_ip2')
    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip2, fakemol_nuc, fakemol)
    dv_g = numpy.einsum('n,xng->gx', atom_charges, v_ng_ip2)
    dv_g = numpy.einsum('gx,g->gx', dv_g, q_sym)
    de -= numpy.asarray([numpy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])
    
    ## --------------- response from stiffness matrices ----------------
    gridslice = pcmobj.surface['gslice_by_atom']
    dF, dA = get_dF_dA(pcmobj.surface)
    
    with_D = pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE'
    dD, dS, dSii = get_dD_dS(pcmobj.surface, dF, with_D=with_D, with_S=True)

    if pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE':
        DA = D*A

    epsilon = pcmobj.eps
    
    #de_dF = v0 * -dSii_dF * q
    #de += 0.5*numpy.einsum('i,inx->nx', de_dF, dF)
    # dQ = v^T K^-1 (dR - dK K^-1 R) v
    if pcmobj.method.upper() == 'C-PCM' or pcmobj.method.upper() == 'COSMO':
        # dR = 0, dK = dS
        de_dS = numpy.einsum('i,ijx,j->ix', vK_1, dS, q)
        de -= numpy.asarray([numpy.sum(de_dS[p0:p1], axis=0) for p0,p1, in gridslice])
        de -= 0.5*numpy.einsum('i,ijx,i->jx', vK_1, dSii, q)
    
    elif pcmobj.method.upper() == 'IEF-PCM' or pcmobj.method.upper() == 'SS(V)PE':
        # IEF-PCM and SS(V)PE formally are the same in gradient calculation
        # dR = f_eps/(2*pi) * (dD*A + D*dA), 
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * numpy.einsum('i,ijx,j->ix', vK_1, dD, Av)
        de_dR -= 0.5*fac * numpy.einsum('i,ijx,j->jx', vK_1, dD, Av)
        de_dR  = numpy.asarray([numpy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dR += 0.5*fac * numpy.einsum('i,ij,jnx,j->nx', vK_1, D, dA, v_grids)
        
        de_dS0  = 0.5*numpy.einsum('i,ijx,j->ix', vK_1, dS, q)
        de_dS0 -= 0.5*numpy.einsum('i,ijx,j->jx', vK_1, dS, q)
        de_dS0  = numpy.asarray([numpy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dS0 += 0.5*numpy.einsum('i,inx,i->nx', vK_1, dSii, q)
        
        vK_1_DA = numpy.dot(vK_1, DA)
        de_dS1  = 0.5*numpy.einsum('j,jkx,k->jx', vK_1_DA, dS, q)
        de_dS1 -= 0.5*numpy.einsum('j,jkx,k->kx', vK_1_DA, dS, q)
        de_dS1  = numpy.asarray([numpy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])
        de_dS1 += 0.5*numpy.einsum('j,jnx,j->nx', vK_1_DA, dSii, q)

        Sq = numpy.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*numpy.einsum('i,ijx,j->ix', vK_1, dD, ASq)
        de_dD -= 0.5*numpy.einsum('i,ijx,j->jx', vK_1, dD, ASq)
        de_dD  = numpy.asarray([numpy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = numpy.dot(vK_1, D)
        de_dA = 0.5*numpy.einsum('j,jnx,j->nx', vK_1_D, dA, Sq)

        de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
        de += de_dR - de_dK
    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")
    
    return de
        
def make_grad_object(grad_method):
    '''
    return solvent gradient object
    '''
    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        def kernel(self, *args, dm=None, atmlst=None, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)
            
            self.de_solvent = grad_kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent
            
            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)

class PCM(ddcosmo.DDCOSMO):
    def __init__(self, mol):
        ddcosmo.DDCOSMO.__init__(self, mol)
        self.method = 'C-PCM'
        self.vdw_scale = 1.2 # default value in qchem
        self.surface = {}
        self._intermediates = {}

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'ddPCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'lmax = %s'         , self.lmax)
        logger.info(self, 'eta = %s'          , self.eta)
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.debug2(self, 'radii_table %s', self.radii_table)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        self.grids.dump_flags(verbose)
        return self

    def build(self, ng=None):
        vdw_scale = self.vdw_scale
        self.radii_table = vdw_scale * Bondi
        mol = self.mol
        if ng is None: 
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]
        
        self.surface = gen_surface(mol, ng=ng, vdw_scale=vdw_scale)
        self._intermediates = {}
        F, A = get_F_A(self.surface)
        D, S = get_D_S(self.surface, with_S=True, with_D=True)
        
        epsilon = self.eps
        if self.method.upper() == 'C-PCM':
            f_epsilon = (epsilon-1.)/epsilon
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0)
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'IEF-PCM':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)            
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

    def _get_vind(self, dms):
        if not self._intermediates or self.grids.coords is None:
            self.build()

        mol = self.mol
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        n_dm = dms.shape[0]

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = self._get_v(self.surface, dms)
        b = numpy.dot(R, v_grids)
        q = numpy.linalg.solve(K, b)

        vK_1 = numpy.linalg.solve(K.T, v_grids)
        q_sym = (q + numpy.dot(R.T, vK_1))/2.0

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym, v_grids)
        
        self._intermediates['K'] = K
        self._intermediates['R'] = R
        self._intermediates['q'] = q
        self._intermediates['q_sym'] = q_sym
        self._intermediates['v_grids'] = v_grids

        return epcm, vmat

    def _get_v(self, surface, dms):
        '''
        return electrostatic potential on surface
        '''
        mol = self.mol
        nao = dms.shape[-1]
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()
        grid_coords = surface['grid_coords']
        exponents   = surface['charge_exp']

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        ngrids = grid_coords.shape[0]
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        v_grids_e = numpy.empty(ngrids)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            v_grids_e[p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[0])

        int2c2e = mol._add_suffix('int2c2e')

        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        v_grids_n = numpy.dot(atom_charges, v_ng)
        
        v_grids = v_grids_n - v_grids_e
        return v_grids

    def _get_vmat(self, q):
        mol = self.mol
        nao = mol.nao
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        ngrids = grid_coords.shape[0]
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        vmat = numpy.zeros([nao,nao])
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            vmat += -numpy.einsum('ijL,L->ij', v_nj, q[p0:p1])
        return vmat
    
    def nuc_grad_method(self, grad_method):
        return make_grad_object(grad_method)

