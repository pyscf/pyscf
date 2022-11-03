#
#
import numpy as np
from pyscf import gto, scf, mp
from pyscf.data.nist import BOHR
from pyscf.data import radii
from pyscf.lib import logger
import sys
'''
From Bayly, Cieplak, Cornell, and Kollman, J. Phys. Chem. 1993, 97, 10269.
Section: Methods

"The charge fitting process begins with having the QM ESP Vi evaluated for each point i of a set of points (grid) fixed in space in the solvent-accessible region around the molecule. The points must lie outside the van der Waals radius of the molecule ..."

Procedure:
1) esp_grid : grid points are genrated.
2) esp_esp : Electrostatic potential (ESP) per each grid is estimated.
3) esp_fit : the charge fitting procedure.
           default penalty function is a hyperbolic form (see Eq. (9)) with a target charge of zero (q0 = 0),
           since we cannot assign the target charge for each atomic site by hand.
'''

ang2bohr = 1.0/BOHR


'''
The values in 'options' (in esp_atomic_charges) are set to reproduce the NWCHEM ESP calculation.
--- <input>---

charge    0
geometry units angstroms noautosym noautoz nocenter
    O   0.00000      -0.07579       0.0000
    H   0.86681       0.60144       0.0000
    H  -0.86681       0.60144       0.0000
end

basis spherical
 * library "aug-cc-pvdz"
end

scf
 thresh 1e-8
 print low
end

task scf       energy

esp
  recalculate
  restrain hfree
end

task esp
---<end input>-----

--- (output: RHF/aug-cc-pvdz)--
    Atom              Coordinates                           Charge

                                                  ESP         RESP        RESP2
                                                                         constr

    1 O     0.000000   -0.007579    0.000000   -0.673531   -0.672911   -0.672911
    2 H     0.086681    0.060144    0.000000    0.333545    0.333230    0.333230
    3 H    -0.086681    0.060144    0.000000    0.339986    0.339681    0.339681
----(end output)----------
'''

def get_esp_radii (probe):
    ''' Obtain Solvent Inaccessible Radii

    Parameters
    ----------
    probe: float (in A)
       A radius in A determining the envelope around the molecule
    Returns
    ------
    ESP_RADII: np.array(float)
    '''

    ESP_RADII = ang2bohr * np.array (
        (0, # Ghost atom
         0.30,                                     1.22, # 1s
         1.23, 0.89, 0.88, 0.77, 0.70, 0.66, 0.58, 1.60, # 2s2p
         1.40, 1.36, 1.25, 1.17, 1.10, 1.04, 0.99, 1.91, # 3s3p
         2.03, 1.74,                                     # 4s (K, Ca)
         1.44, 1.32, 1.22, 1.19, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25, # 3d (Sc,.., Zn)
         1.25, 1.22, 1.21, 1.17, 1.14, 1.98, # 4p (Ga, .., Kr)
         2.22, 1.92,                                     # 5s (Rb, Sr)
         1.62, 1.45, 1.34, 1.29, 1.27, 1.24, 1.25, 1.28, 1.34, 1.41, # 4d (Y,..,Cd)
         1.50, 1.40, 1.41, 1.37, 1.33, 2.09, # 5p (In,.., Xe)
         2.35, 1.98                                      # 6s
        ))


    prob_radius = probe*ang2bohr
    ESP_RADII += prob_radius

    return ESP_RADII


def esp_grid (mol, rcut=3.0, space=0.5, probe=0.7):

    ''' Generate grid points as the first step

    Parameters
    ----------
    mol : gto.Mole() : provides the coordinates and atomic numbers.
    rcut: float
         A cut-off distance in A for the solvent accessible region around the molecule
    space: float
         A grid spacing in A for the regularly spaced grid points
    probe : float
         A radius in A determining the envolope around the molecule

    Returns
    -------
    grids : np.array (float)
    '''

    #ESP_RADII = radii.VDW*1.2 #get_esp_radii (probe)
    ESP_RADII = get_esp_radii (probe)
    qm_xyz = mol.atom_coords()
    qm_znum= mol.atom_charges ()
    natom  = qm_znum.shape[0]

    grid_min = np.array ( [min(qm_xyz[:,0]),min(qm_xyz[:,1]),min(qm_xyz[:,2])])
    grid_max = np.array ( [max(qm_xyz[:,0]),max(qm_xyz[:,1]),max(qm_xyz[:,2])])


    ##
    ## calculate the grid size
    ##

    _rcut  = rcut*ang2bohr
    _rcut2 = _rcut*_rcut
    _spac  = space*ang2bohr

    ngrid_x = int( (grid_max[0] - grid_min[0] + 2.0*_rcut) / _spac) + 1
    ngrid_y = int( (grid_max[1] - grid_min[1] + 2.0*_rcut) / _spac) + 1
    ngrid_z = int( (grid_max[2] - grid_min[2] + 2.0*_rcut) / _spac) + 1

    small = 1.0e-8
    grids = []

    for iz in range (ngrid_z):
        for iy in range (ngrid_y):
            for ix in range (ngrid_x):
                gv = grid_min - _rcut + _spac *np.array ([ix, iy, iz])

                rmin2 = _rcut2
                lupdate = True

                for ia in range (natom):
                    znum = qm_znum[ia]
                    rad  = ESP_RADII[znum]
                    rad2 = rad*rad

                    dr = gv - qm_xyz[ia]
                    r2 = np.einsum ('i,i', dr, dr)

                    if rad2 - r2 > small:
                        lupdate = False
                        break

                    if rmin2 - r2 > small:
                        rmin2 = r2

                if lupdate:
                    if _rcut2 - rmin2 > small:
                        grids.append (list(gv))

    grids = np.array (grids)

    return grids


def esp_esp (mol, dm, grids):
    ''' Estimate the electrostatic potential (ESP) at each grid point and atomic site as the second step.

    Parameters
    ----------
    mol = pyscf.Mole()
    dm  = density Matrix
    grids = np.array (ngrid, 3) : grid point

    Returns
    -------
    g_vals + z_vals : np.array : the ESP value at each grid
    '''

    nsize      = grids.shape[0]
    qm_xyz     = mol.atom_coords()
    qm_charges = mol.atom_charges()
    natom      = qm_charges.shape[0]

    drg = qm_xyz[:, None, :] - grids # (Natom, Ngrid, 3)
    dr  = np.linalg.norm (drg, axis=2) # (Natom, Ngrid)

    '''
     *** The electrostatic potential at the atomic center
      z_val = \sum_i \frac{Z_i}{|R_i - r|}
    '''

    z_val = np.einsum ('ig, i->g', 1.0/dr, qm_charges)

    '''
    *** The electrostatic potential at a grid point (g)
     g_val(g) = -\sum_{ij} D_{ij} V_{ij} (g),
     where V_{ij} (g) = \int \frac{\psi_i (r) \psi_j (r)}{|r - g|} dr
     g is the grid point
     r is the position of the one-electron
     \psi is a basis of (Gaussian) atomic orbitals.
    '''

    g_val = []
    for gv in grids:
        with mol.with_rinv_origin (gv):
            v = mol.intor ('int1e_rinv')

        gval = -np.einsum ('ij,ij', dm, v)

        g_val.append (gval)

    return  np.array (g_val) + z_val



def esp_fit (mol, grids, grids_val,
             restraint, hfree, resp_a, resp_b, maxiter, tolerance, verbose):
    ''' Fitting procedure for (R)ESP atomic charges
        In case of RESP, a penalty function is a hyperbolic form:
        penalty funciton = resp_a \sum_i (q_i*q_i - resp_b*resp_b)**0.5 - resp_b)

    Parameters:
    ----------
    mol : pyscf.Mole()
    grids : np.array (ngrid, 3), grid coordinates
    grids_val : np.array (ngrid), the V at grid point
    restraint : bool
        whether restraint included
    hfree  : bool
         whether hydrogen atoms excluded or included in restraint
    resp_a : float
         restraint scale
    resp_b : float
         restraint parabola tightness
    maxiter : int
         maximum number of interactions
    tolerance : float,
         tolerance for charges in the fitting
    verbose : int
         print out message

    Returns:
    qf : list
       (R)ESP atomic charges
    '''

    qm_xyz  = mol.atom_coords()
    qm_znum = mol.atom_charges()

    natoms = qm_xyz.shape[0]
    ndim   = natoms + 1
    am = np.zeros ((ndim, ndim))

    drg = qm_xyz [:,None,:] - grids    # (Natom, Ngrid, 3)
    dr  = np.linalg.norm (drg, axis=2) # (Natom, Ngrid)
    am[:natoms,:natoms] = np.einsum ('ig, jg->ij', 1.0/dr, 1.0/dr)
    am[:natoms,natoms] = 1.0
    am[natoms,:natoms] = 1.0

    ### construct column vector b
    bv = np.zeros ( (ndim) )

    bv[:natoms] = np.einsum ('ig, g->i', 1.0/dr, grids_val)
    bv[natoms] = mol.charge

    am_inv = np.linalg.inv(am)
    qf = np.einsum ('ij,j->i', am_inv, bv)

    if verbose > logger.QUIET:
        print ('VERBOSE: ESP atomic charges: ', qf[:natoms])

    if restraint:

        # start RESP
        qf_keep = np.copy(qf)
        am_keep = np.copy(am)

        niter = 0
        while niter < maxiter:
            niter += 1

            am = np.copy(am_keep)

            '''
            Hyperbolic Restraints
            Eq(14) is modified into
            Bj = sum_i {V_i/r_{ij}}
            since q0_j is zero.
            '''
            for ia in range (natoms):
                if (not hfree) or qm_znum[ia] != 1:
                    am[ia,ia] = am_keep[ia, ia] + resp_a/np.sqrt(qf[ia]*qf[ia] + resp_b*resp_b)

            am_inv = np.linalg.inv (am)

            difm = 0.0
            for ia in range (natoms):
                vsum = 0.0
                for jb in range (ndim):
                    vsum += am_inv[ia, jb]*bv[jb]
                qf[ia] = vsum

                dif = (vsum - qf_keep[ia])**2
                if difm < dif:
                    difm = dif

            difm = np.sqrt (difm)
            qf_keep = np.copy(qf)

            if (difm < tolerance):
                break

        if verbose > logger.QUIET:
            print ('VERBOSE: RESP atomic charges: ', qf[:natoms])

    return qf[:natoms]


def esp_atomic_charges (mol, dm, options_dict={}, verbose=0):
    ''' Estimate (R)ESP atomic charges

    Parameters
    ----------
    mol : pyscf.gto.Mole()
    dm  : density matrix
    options_dict{} : dict, optional
         dictionary of user's defined options
    verbose : 0, optional

    Returns
    -------
    charges : list
    '''

    options = {
        # A cut-off distance for the solvent accessible region around the molecule
        "RCUT"  : 3.0, # Angstrom
        # A grid spacing for the regularly spaced grid points
        "SPACE" : 0.5, # Angstrom
        # A radius determining the envelope around the molecule
        "PROBE" : 0.7, # Angstrom
        "RESTRAINT" : True,
        # Exclude hydrogen atoms from the restaining procedure.
        "RESP_HFREE" : True,
        "RESP_A"  : 0.001, # au
        "RESP_B"  : 0.1,   # au
        "RESP_MAXITER" : 25,
        "RESP_TOLERANCE" : 1.0e-4, # e
    }

    for key in options_dict.keys():
        key_upper = key.upper()
        if key_upper in options.keys():
            options[key_upper] = options_dict[key]

    grids = esp_grid (mol,
                      options['RCUT'],
                      options['SPACE'],
                      options['PROBE'])

    grids_val = esp_esp (mol, dm, grids)

    esp_chg = esp_fit (mol, grids, grids_val,
                       options['RESTRAINT'],
                       options['RESP_HFREE'],
                       options['RESP_A'],
                       options['RESP_B'],
                       options['RESP_MAXITER'],
                       options['RESP_TOLERANCE'],
                       verbose)

    return esp_chg


#
# For post-HF methods, the response of HF orbitals needs to be considered in
# the analytical gradients. It is similar to the gradients code implemented in
# the module pyscf.grad.
#
# Below we use MP2 gradients as example to demonstrate how to include the
# orbital response effects in the force for MM particles.
#

# Based on the grad_elec function in pyscf.grad.mp2
def make_rdm1_with_orbital_response(mp):
    import time
    from pyscf import lib
    from pyscf.grad.mp2 import _response_dm1, _index_frozen_active, _shell_prange
    from pyscf.mp import mp2
    from pyscf.ao2mo import _ao2mo
    from functools import reduce

    log = lib.logger.new_logger(mp)
    time0 = time.clock(), time.time()
    mol = mp.mol

    log.debug('Build mp2 rdm1 intermediates')
    d1 = mp2._gamma1_intermediates(mp, mp.t2)
    doo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)

    with_frozen = not (mp.frozen is None or mp.frozen is 0)
    OA, VA, OF, VF = _index_frozen_active(mp.get_frozen_mask(), mp.mo_occ)
    orbo = mp.mo_coeff[:,OA]
    orbv = mp.mo_coeff[:,VA]
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]

# Partially transform MP2 density matrix and hold it in memory
# The rest transformation are applied during the contraction to ERI integrals
    part_dm2 = _ao2mo.nr_e2(mp.t2.reshape(nocc**2,nvir**2),
                            np.asarray(orbv.T, order='F'), (0,nao,0,nao),
                            's1', 's1').reshape(nocc,nocc,nao,nao)
    part_dm2 = (part_dm2.transpose(0,2,3,1) * 4 -
                part_dm2.transpose(0,3,2,1) * 2)

    offsetdic = mol.offset_nr_by_atom()
    diagidx = np.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    Imat = np.zeros((nao,nao))

# 2e AO integrals dot 2pdm
    max_memory = max(0, mp.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = lib.einsum('pi,iqrj->pqrj', orbo[ip0:ip1], part_dm2)
            dm2buf+= lib.einsum('qi,iprj->pqrj', orbo, part_dm2[:,ip0:ip1])
            dm2buf = lib.einsum('pqrj,sj->pqrs', dm2buf, orbo)
            dm2buf = dm2buf + dm2buf.transpose(0,1,3,2)
            dm2buf = lib.pack_tril(dm2buf.reshape(-1,nao,nao)).reshape(nf,nao,-1)
            dm2buf[:,:,diagidx] *= .5

            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None
            dm2buf = None
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

# Recompute nocc, nvir to include the frozen orbitals and make contraction for
# the 1-particle quantities, see also the kernel function in ccsd_grad module.
    mo_coeff = mp.mo_coeff
    mo_energy = mp._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = np.count_nonzero(mp.mo_occ > 0)
    Imat = reduce(np.dot, (mo_coeff.T, Imat, mp._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = np.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mp._scf.get_veff(mp.mol, dm1) * 2
    Xvo = reduce(np.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mp, Xvo)

    # Transform to AO basis
    dm1 = reduce(np.dot, (mo_coeff, dm1mo, mo_coeff.T))
    dm1 += mp._scf.make_rdm1(mp.mo_coeff, mp.mo_occ)
    return dm1


if __name__ == '__main__':

    # XYZ in Bohr
    qm_atm_list = [ ['O', np.array(( 0.00000,      -0.07579,       0.00000))*ang2bohr],
                    ['H', np.array(( 0.86681,       0.60144,       0.00000))*ang2bohr],
                    ['H', np.array((-0.86681,       0.60144,       0.00000))*ang2bohr] ]

    #

    mol = gto.Mole()
    #mol.basis = '6-311g**'
    mol.basis = 'aug-cc-pvdz'
    mol.atom  = qm_atm_list
    mol.charge = 0
    mol.unit = 'Bohr'
    mol.build ()

    # RHF
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf = mf.run(verbose=0)

    esp_options = {
        "probe"   : 0.7, # A
        "restraint" : True, # False: ESP atomic charges, True: RESP atomic charges
        "resp_hfree" : True, # Exclude hydrogen atoms from the restaining procedure.
        "resp_a"  : 0.001, # au
        "resp_b"  : 0.1,   # au
        "resp_maxiter" : 25, # maximum iteraction
        "resp_tolerance" : 1.0e-4, # e
    }

    #-- (R)ESP CHARGE at RHF
    # RHF Density Matrix
    dm = mf.make_rdm1()
    print ('With RHF density matrix')
    esp_chg = esp_atomic_charges (mol, dm, esp_options, verbose=1)
    print ('esp_chg at RHF', esp_chg)


    m = mp.MP2 (mf,frozen=1).run (verbose=0)
    #-- (R)ESP CHARGE at MP2
    # MP2 Density Matrix
    dm = make_rdm1_with_orbital_response (m)
    print ('With MP2 density matrix:')
    esp_chg = esp_atomic_charges (mol, dm, esp_options, verbose=1)

    print ('esp_chg at MP2', esp_chg)
