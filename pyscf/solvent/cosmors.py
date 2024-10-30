# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Provides functionality to generate COSMO files and sigma-profiles
for the further use in COSMO-RS/COSMO-SAC computations and/or ML
'''

#%% Imports

import re as _re
from itertools import product as _product

import numpy as _np

from pyscf import __version__
from pyscf.data.nist import BOHR as _BOHR



#%% SAS volume

def _get_line_atom_intersection(x, y, atom, r):
    '''Returns z-coordinates of boundaries of intersection of
    the (x,y,0) + t(0,0,1) line and vdW sphere

    Arguments:
        x (float): x parameter of the line
        y (float): y parameter of the line
        atom (np.ndarray): xyz-coordinates of the atom
        r (float): van der Waals radii of the atom

    Returns:
        _tp.Optional[_tp.List[float, float]]: z-coordinates
            of the line/sphere intersection, and None
            if they do not intersect

    '''
    # check distance between line and atom
    d = _np.linalg.norm(_np.array([x,y]) - atom[:2])
    if d >= r:
        return None
    # find interval
    dz = (r**2 - d**2)**0.5
    interval = [atom[2] - dz, atom[2] + dz]

    return interval


def _unite_intervals(intervals):
    '''Unites float intervals

    Arguments:
        intervals (_tp.List[_tp.List[float, float]]): list of intervals

    Returns:
        _tp.List[_tp.List[float, float]]: list of united intervals

    '''
    united = []
    for begin, end in sorted(intervals):
        if united and united[-1][1] >= begin:
            united[-1][1] = end
        else:
            united.append([begin, end])

    return united


def _get_line_sas_intersection_length(x, y, coords, radii):
    '''Finds total length of all (x,y,0) + t(0,0,1) line's intervals
    enclosed by intersections with solvent-accessible surface

    Arguments:
        x (float): x parameter of the line
        y (float): y parameter of the line
        atoms (_np.ndarray): (n*3) array of atomic coordinates
        radii (_np.ndarray): array of vdW radii

    Returns:
        float: total length of all line's intervals enclosed by SAS

    '''
    intervals = [_get_line_atom_intersection(x, y, atom, r) for atom, r in zip(coords, radii)]
    intervals = _unite_intervals([_ for _ in intervals if _])
    length = sum([end - begin for begin, end in intervals])

    return length


def get_sas_volume(surface, step=0.2):
    '''Computes volume [A**3] of space enclosed by solvent-accessible surface

    Arguments:
        surface (dict): dictionary containing SAS parameters (mc.with_solvent.surface)
        step (float): grid spacing parameter, [Bohr]

    Returns:
        float: SAS-defined volume of the molecule [A**3]

    '''
    # get parameters
    coords = surface['atom_coords']
    radii = _np.array([surface['R_vdw'][i] for i, j in surface['gslice_by_atom']])
    # compute minimaxes of SAS coordinates
    bounds = _np.array([(coords - radii[:, _np.newaxis]).min(axis = 0),
                        (coords + radii[:, _np.newaxis]).max(axis = 0)])
    # swap axes to minimize integration mesh
    axes = [(val, idx) for idx, val in enumerate(bounds[1] - bounds[0])]
    axes = [idx for val, idx in sorted(axes)]
    coords = coords[:,axes]
    bounds = bounds[:,axes]
    # mesh parameters
    xs = _np.linspace(bounds[0,0], bounds[1,0],
                      int(_np.ceil((bounds[1,0] - bounds[0,0])/step)) + 1)
    ys = _np.linspace(bounds[0,1], bounds[1,1],
                      int(_np.ceil((bounds[1,1] - bounds[0,1])/step)) + 1)
    dxdy = (xs[1] - xs[0])*(ys[1] - ys[0])
    # integration
    V = sum([dxdy * _get_line_sas_intersection_length(x, y, coords, radii) for x, y in _product(xs, ys)])

    return V * _BOHR**3



#%% COSMO-files

def _check_feps(mf, ignore_low_feps):
    '''Checks f_epsilon value and raises ValueError for f_eps < 1

    Arguments:
        mf: processed SCF with PCM solvation
        ignore_low_feps (bool): if True, does not raise ValueError if feps < 1

    Raises:
        ValueError: if f_epsilon < 1 and ignore_low_feps flag is set to False

    '''
    if ignore_low_feps:
        return

    f_eps = mf.with_solvent._intermediates['f_epsilon']
    if f_eps < 1.0:
        message  = f'Low f_eps value: {f_eps}. '
        message += 'COSMO-RS requires f_epsilon=1.0 or eps=float("inf"). '
        message += 'Rerun computation with correct epsilon or use the ignore_low_feps argument.'
        raise ValueError(message)

    return


def _lot(mf):
    '''Returns level of theory in method-disp/basis/solvation format (raw solution)

    Arguments:
        mf: processed SCF

    Returns:
        str: method-dispersion/basis/solvation

    '''
    # method
    method = mf.xc if hasattr(mf, 'xc') else None
    if method is None:
        match = _re.search('HF|MP2|CCSD', str(type(mf)))
        method = match.group(0) if match else '???'
    # dispersion
    match = _re.search('D3|D4', str(type(mf)))
    disp = '-' + match.group(0) if match else ''
    # other
    basis = mf.mol.basis
    solv = '/' + mf.with_solvent.method if hasattr(mf, 'with_solvent') else ''
    # approx
    lot = f'{method}{disp}/{basis}{solv}'.lower()

    return lot


def write_cosmo_file(fout, mf, step=0.2, ignore_low_feps=False):
    '''Saves COSMO file

    Arguments:
        fout: writable file object
        mf: processed SCF with PCM solvation
        step (float): grid spacing parameter to compute volume, [Bohr]
        ignore_low_feps (bool): if True, does not raise ValueError if feps < 1

    Raises:
        ValueError: if f_epsilon < 1 and ignore_low_feps flag is set to False

    '''
    _check_feps(mf, ignore_low_feps)

    # qm params
    fout.write('$info\n')
    fout.write(f'PySCF v. {__version__}, {_lot(mf)}\n')

    # cosmo data
    f_epsilon = mf.with_solvent._intermediates['f_epsilon']
    n_segments = len(mf.with_solvent._intermediates['q'])
    area = sum(mf.with_solvent.surface['area'])
    volume = get_sas_volume(mf.with_solvent.surface, step) /  _BOHR**3
    # print
    fout.write('$cosmo_data\n')
    fout.write(f'  fepsi  = {f_epsilon:>.8f}\n')
    fout.write(f'  nps    = {n_segments:>10d}\n')
    fout.write(f'  area   = {area:>10.2f}       # [Bohr**2]\n')
    fout.write(f'  volume = {volume:>10.2f}       # [Bohr**3]\n')

    # atomic coordinates
    atoms = range(1, 1 + len(mf.mol.elements))
    xs, ys, zs = zip(*mf.mol.atom_coords().tolist())
    elems = [elem.lower() for elem in mf.mol.elements]
    radii = [mf.with_solvent.surface['R_vdw'][i] for i, j in mf.with_solvent.surface['gslice_by_atom']]
    # print
    fout.write('$coord_rad\n')
    fout.write('#atom   x [Bohr]           y [Bohr]           z [Bohr]      element  radius [Bohr]\n')
    for atom, x, y, z, elem, r in zip(atoms, xs, ys, zs, elems, radii):
        fout.write(f'{atom:>4d}{x:>19.14f}{y:>19.14f}{z:>19.14f}  {elem:<2}{r:>12.5f}\n')

    # cosmo parameters
    screening_charge = sum(mf.with_solvent._intermediates['q'])
    E_tot = sum(mf.scf_summary.values())
    E_diel = mf.scf_summary['e_solvent']
    # print
    fout.write('$screening_charge\n')
    fout.write(f'  cosmo                          = {screening_charge:>15.6f}\n')
    fout.write('$cosmo_energy\n')
    fout.write(f'  Total energy [a.u.]            = {E_tot:>19.10f}\n')
    fout.write(f'  Dielectric energy [a.u.]       = {E_diel:>19.10f}\n')

    # segments
    xs, ys, zs = zip(*mf.with_solvent.surface['grid_coords'].tolist())
    ns = range(1, len(xs) + 1)
    atoms = []
    for idx, (start, end) in enumerate(mf.with_solvent.surface['gslice_by_atom']):
        atoms += [idx + 1] * (end - start)
    qs = mf.with_solvent._intermediates['q']
    areas = mf.with_solvent.surface['area'] * _BOHR**2
    sigmas = qs / areas
    pots = mf.with_solvent._intermediates['v_grids']
    # print legend
    fout.write('$segment_information\n')
    fout.write('# n             - segment number\n')
    fout.write('# atom          - atom associated with segment n\n')
    fout.write('# x, y, z       - segment coordinates, [Bohr]\n')
    fout.write('# charge        - segment charge\n')
    fout.write('# area          - segment area [A**2]\n')
    fout.write('# charge/area   - segment charge density [e/A**2]\n')
    fout.write('# potential     - solute potential on segment\n')
    fout.write('#\n')
    fout.write('#  n   atom              position (X, Y, Z)                   ')
    fout.write('charge         area        charge/area     potential\n')
    fout.write('#\n')
    fout.write('#\n')
    # print params
    for n, x, y, z, atom, q, area, sigma, pot in zip(ns, xs, ys, zs, atoms, qs, areas, sigmas, pots):
        line  = f'{n:>5d}{atom:>5d}{x:>15.9f}{y:>15.9f}{z:>15.9f}'
        line += f'{q:>15.9f}{area:>15.9f}{sigma:>15.9f}{pot:>15.9f}\n'
        fout.write(line)

    return



#%% Sigma-profile

def get_sigma_profile(mf, sigmas_grid, ignore_low_feps=False):
    '''Computes sigma-profile in [-0.025..0.025] e/A**2 range as in
    https://github.com/usnistgov/COSMOSAC/blob/master/profiles/to_sigma.py#L181
    https://doi.org/10.1021/acs.jctc.9b01016

    Arguments:
        mf: processed SCF with PCM solvation
        sigmas_grid (_np.ndarray): grid of screening charge values,
            e.g. np.linspace(-0.025, 0.025, 51)
        ignore_low_feps (bool): if True, does not raise ValueError if feps < 1

    Returns:
        _np.ndarray: array of 51 elements corresponding to the p(sigma) values for
            the screening charge values in [-0.025..0.025] e/A**2 range

    Raises:
        ValueError: if f_epsilon < 1 and ignore_low_feps flag is set to False

    '''
    _check_feps(mf, ignore_low_feps)

    # prepare params
    qs = mf.with_solvent._intermediates['q']
    areas = mf.with_solvent.surface['area'] * _BOHR**2
    sigmas = qs / areas
    step = sigmas_grid[1] - sigmas_grid[0]
    max_sigma = sigmas_grid[-1] + step
    n_bins = len(sigmas_grid)

    # compute profile
    psigma = _np.zeros(n_bins)
    for sigma, area in zip(sigmas, areas):
        # get index of left sigma grid value
        left = int(_np.floor((sigma - sigmas_grid[0]) / step))
        if left < -1 or left > n_bins - 1:
            continue
        # get impact of the segment to the left bin
        val_right = sigmas_grid[left+1] if left < n_bins - 1 else max_sigma
        w_left = (val_right - sigma) / step
        # add areas to p(sigma)
        if left > -1:
            psigma[left] += area * w_left
        if left < n_bins - 1:
            psigma[left+1] += area * (1 - w_left)

    return psigma


def get_cosmors_parameters(mf, sigmas_grid=_np.linspace(-0.025, 0.025, 51),
                           step=0.2, ignore_low_feps=False):
    '''Computes main COSMO-RS parameters

    Arguments:
        mf: processed SCF with PCM solvation
        step (float): grid spacing parameter to compute volume, [Bohr]
        ignore_low_feps (bool): if True, does not raise ValueError if feps < 1

    Returns:
        dict: main COSMO-RS parameters, including sigma-profile, volume, surface area,
            SCF and dielectric energies

    Raises:
        ValueError: if f_epsilon < 1 and ignore_low_feps flag is set to False

    '''
    _check_feps(mf, ignore_low_feps)

    # get params
    lot = _lot(mf)
    area = sum(mf.with_solvent.surface['area']) * _BOHR**2
    volume = get_sas_volume(mf.with_solvent.surface, step)
    psigma = get_sigma_profile(mf, sigmas_grid, ignore_low_feps=True)
    E_tot = sum(mf.scf_summary.values())
    E_diel = mf.scf_summary['e_solvent']

    # output
    params = {'PySCF version': __version__,
              'Level of theory': lot,
              'Surface area, A**2': float(area), # since np.float is not json-seriazable
              'Volume, A**3': float(volume),
              'Total energy, a.u.': float(E_tot),
              'Dielectric energy, a.u.': float(E_diel),
              'Screening charge density, A**2': psigma.tolist(),
              'Screening charge, e/A**2': sigmas_grid.tolist()}

    return params


