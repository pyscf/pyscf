import sys
import tempfile

from functools import reduce
import numpy
import numpy as np
import scipy.linalg
import h5py
import ctypes
import copy

from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import diis
from pyscf.scf import _vhf
from pyscf.scf import chkfile
from pyscf.data import nist
from pyscf import __config__
from pyscf.lib import param
from pyscf.data import elements
from pyscf.scf import hf, rohf, addons


##### generate more accurate initial guess for the density matrix

def get_atm_nrhf(mol, 
                 atm_config=None,
                 # atomic_configuration=elements.NRSRHF_CONFIGURATION
                 ):
    
    # get the modified atomic configuration
    
    atomic_configuration = copy.deepcopy(elements.NRSRHF_CONFIGURATION)
    if atm_config is not None:
        for key in atm_config:
            charge = gto.charge(key)
            atomic_configuration[charge] = atm_config[key]['occ_config']
    
    
    elements_now = set([a[0] for a in mol._atom])
    logger.info(mol, 'Spherically averaged atomic HF for %s', elements_now)

    atm_template = copy.copy(mol)
    atm_template.charge = 0
    atm_template.enuc = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphAverageRHF does not support cartesian basis

    atm_scf_result = {}
    for ia, a in enumerate(mol._atom):
        element = a[0]
        if element in atm_scf_result:
            continue

        # atm = copy.deepcopy(atm_template)
        atm = atm_template
        atm.charge = 0
        if atm_config is not None and element in atm_config:
            # atm.nelectron = sum(atm_config[element]['occ_config'])
            atm.charge = atm_config[element]['charge']
        atm._atom = [a]
        atm._atm = mol._atm[ia:ia+1]
        atm._bas = mol._bas[mol._bas[:,0] == ia].copy()
        atm._ecpbas = mol._ecpbas[mol._ecpbas[:,0] == ia].copy()
        # Point to the only atom
        atm._bas[:,0] = 0
        atm._ecpbas[:,0] = 0
        if element in mol._pseudo:
            atm._pseudo = {element: mol._pseudo.get(element)}
            #raise NotImplementedError
        atm.spin = atm.nelectron % 2

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            mo_occ = mo_energy = numpy.zeros(nao)
            mo_coeff = numpy.zeros((nao,nao))
            atm_scf_result[element] = (0, mo_energy, mo_coeff, mo_occ)
        elif atm._pseudo:
            from pyscf.scf import atom_hf_pp
            atm.a = None
            if atm.nelectron == 1:
                atm_hf = atom_hf_pp.AtomHF1ePP(atm)
            else:
                atm_hf = atom_hf_pp.AtomSCFPP(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)
        else:
            if atm.nelectron == 1:
                atm_hf = AtomHF1e(atm)
            else:
                atm_hf = AtomSphAverageRHF(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)
    return atm_scf_result

def init_guess_by_atom(mol, atm_config=None):
    '''Generate initial guess density matrix from superposition of atomic HF
    density matrix.  The atomic HF is occupancy averaged RHF

    Returns:
        Density matrix, 2D ndarray
    '''
    # from pyscf.scf import atom_hf
    # atm_scf = atom_hf.get_atm_nrhf(mol)
    atm_scf = get_atm_nrhf(mol, atm_config)
    aoslice = mol.aoslice_by_atom()
    atm_dms = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)

        if symb in atm_scf:
            e_hf, e, c, occ = atm_scf[symb]
            if occ.ndim == 2:
                dm = numpy.dot(c[0]*occ[0], c[0].conj().T)
                dm += numpy.dot(c[1]*occ[1], c[1].conj().T)
            else:
                dm = numpy.dot(c*occ, c.conj().T)
        else:  # symb's basis is not specified in the input
            nao_atm = aoslice[ia,3] - aoslice[ia,2]
            dm = numpy.zeros((nao_atm, nao_atm))

        atm_dms.append(dm)

    dm = scipy.linalg.block_diag(*atm_dms)

    if mol.cart:
        cart2sph = mol.cart2sph_coeff(normalized='sp')
        dm = reduce(numpy.dot, (cart2sph, dm, cart2sph.T))

    for k, v in atm_scf.items():
        logger.debug(mol, 'Atom %s, E = %.12g', k, v[0])
    return dm

##### analysis the density matrix

def analysis_dm(mol, dm, distance_matrix):
    '''
    
    analysis the elements of the density matrix
    
    Args: 
        mol : Mole object
        dm  : 2D ndarray, density matrix over atomic orbitals
        distance_matrix : 2D ndarray, distance matrix between atoms (up to lattice translation)
    '''
    
    natm = mol.natm
    aoslice = mol.aoslice_by_atom()

    for i in range(natm):
        slice_i = aoslice[i][2:]
        for j in range(i+1):
            slice_j = aoslice[j][2:]
            dm_ij = dm[slice_i[0]:slice_i[1], slice_j[0]:slice_j[1]]
            # print("dm_ij = ", dm_ij)
            print("atm %2d atm %2d distance %12.5f dm_max %12.5e" % (i, j, distance_matrix[i,j], np.max(np.abs(dm_ij))))

def analysis_dm_on_grid(mydf, dm, distance_matrix):
    '''
    
    analysis the elements of the density matrix on the grid
    
    within this subroutine, density matrix is evaluated on the grid
    
    Args:

        mydf : DF object, must be ISDF object
        dm   : 2D ndarray, density matrix over atomic orbitals
        distance matrix : 2D ndarray, distance matrix between atoms (up to lattice translation)
        
    '''
    
    mol = mydf.cell
    
    aoRg = mydf.aoRg # must be aoR holder 
    aoR = mydf.aoR
    
    natm = mol.natm
    
    naux = mydf.naux
    nao = mydf.nao
    
    ngrid = np.prod(mydf.cell.mesh)
    
    from pyscf.pbc.df.isdf.isdf_linear_scaling_jk import __get_DensityMatrixonRgAO_qradratic
    
    Density_RgAO = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg, mydf.Density_RgAO_buf, False)
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    ddot_buf1 = np.ndarray((naux, max_nao_involved), buffer=mydf.build_k_buf)
    offset = naux * max_nao_involved * ddot_buf1.dtype.itemsize
    pack_buf = np.ndarray((naux, max_nao_involved), buffer=mydf.build_k_buf, offset=offset)
    offset += pack_buf.size * pack_buf.dtype.itemsize
    ddot_buf2 = np.ndarray((naux, max_ngrid_involved), buffer=mydf.build_k_buf, offset=offset)

    libpbc = lib.load_library('libpbc')
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None

    N_elmt_statistics = [0, 0, 0, 0, 0, 0 ,0 ,0, 0] # >1e-3, >1e-4, >1e-5, >1e-6, > 1e-7, >1e-8, >1e-9, >1e-10, < 1e-10

    ngrid_loc = 0
    for atm_j, aoR_holder in enumerate(aoR):
            
        if aoR_holder is None:
            continue
            
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_invovled = aoR_holder.aoR.shape[0]
            
        #### pack the density matrix ####
            
        if nao_invovled == nao:
            Density_RgAO_packed = Density_RgAO
        else:
            # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
            Density_RgAO_packed = np.ndarray((naux, nao_invovled), buffer=pack_buf)
            fn_packcol1(
                Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao_invovled),
                Density_RgAO.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
            
        # V_tmp = Density_RgR[:, ngrid_loc:ngrid_loc+ngrid_now] * V_R[:, ngrid_loc:ngrid_loc+ngrid_now]
            
        ddot_res2 = np.ndarray((naux, ngrid_now), buffer=ddot_buf2)
        lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res2)
        Density_RgR = ddot_res2
        
        ngrid_bra = 0
        for atm_i, aoR_holder_bra in enumerate(aoRg):
            ngrid_bra_now = aoR_holder_bra.aoR.shape[1]
            
            Density_RgR_tmp = Density_RgR[ngrid_bra:ngrid_bra+ngrid_bra_now, :]
            
            max_tmp = np.max(np.abs(Density_RgR_tmp))
            
            print("atm %2d atm %2d distance %12.5f dm_grid_max %12.5e dm_grid_avg = %12.5e" % (atm_i, atm_j, distance_matrix[atm_i, atm_j], max_tmp, np.mean(np.abs(Density_RgR_tmp))))
            # where = np.where(np.abs(Density_RgR_tmp) < 1e-8)
            ngrid_bra += ngrid_bra_now
        
            #### statistics ####
            
            for i in range(9):
                N_elmt_statistics[i] += np.sum(np.abs(Density_RgR_tmp) > 1e-3 * 10**(-i))
    
    ### print statistics ###
    
    for i, data in enumerate(N_elmt_statistics):
        print("nelmt abs > 1e-%d = %d, portion = %.2e" % (i+3, data, float(data) / naux / ngrid))
        

if __name__ == '__main__':

    from pyscf.lib.parameters import BOHR
    import pyscf.pbc.df.isdf.isdf_k as ISDF_K

    # test get_atm_nrhf 
    
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu',	(1.927800,	1.927800,	1.590250)],
['O',	(1.927800,	0.000000,	1.590250)],
['O',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
    ]
    basis = {
        'Cu':'ecpccpvdz', 'O': 'ecpccpvdz', 'Ca':'ecpccpvdz'
    }
    pseudo = {'Cu': 'gth-pbe-q19', 'O': 'gth-pbe', 'Ca': 'gth-pbe'}
    ke_cutoff = 128 
    prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    
    cell = prim_cell
    
    atm_config = {
        'Cu': {'charge': 2, 'occ_config': [6,12,9,0]},
        'O': {'charge': -2, 'occ_config': [4,6,0,0]},
        'Ca': {'charge': 2, 'occ_config': [6,12,0,0]},
    }
    
    res = get_atm_nrhf(cell, atm_config)
    
    init_guess = init_guess_by_atom(cell, atm_config)
    
    print("init_guess = ", init_guess)
    
    # print(res)
    # print(cell.aoslice_by_atom())