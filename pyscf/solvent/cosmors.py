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
#
# Author: Ivan Chernyshov <ivan.chernyshov@gmail.com>
#

'''
Provides functionality to generate COSMO-files for COSMO-RS computations
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
    '''Computes volume [a.u.] of space enclosed by solvent-accessible surface

    Arguments:
        surface (dict): dictionary containing SAS parameters (mc.with_solvent.surface)
        step (float): grid spacing parameter, [Bohr]

    Returns:
        float: SAS-defined volume of the molecule [a.u.]

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

    return V



#%% COSMO-files

def _lot(mf):
    '''Returns level of theory in method-disp/basis/solvation format (raw solution)

    Arguments:
        mf: processed SCF

    Returns:
        str: method-dispersion/basis/solvation

    '''
    lot = {
        'method': None,
        'basis': None,
        'dispersion': None,
        'solvation_model': None
    }

    # method
    method = mf.xc if hasattr(mf, 'xc') else None
    if method is None:
        match = _re.search('HF|MP2|CCSD', str(type(mf)))
        method = match.group(0) if match else None
    lot['method'] = method

    # basis
    lot['basis'] = mf.mol.basis

    # dispersion
    match = _re.search('D3|D4', str(type(mf)))
    disp = match.group(0) if match else None
    lot['dispersion'] = disp

    # solvation
    solv = mf.with_solvent.method if hasattr(mf, 'with_solvent') else ''
    lot['solvation_model'] = solv

    return lot



def get_pcm_parameters(mf, step=0.2):
    '''Returns a dictionary containing the main PCM computation parameters.
    All physical parameters are expressed in atomic units (a.u.).
    Please also note, that outlying charge correction is not implemented yet.

    Arguments:
        mf: processed SCF with PCM solvation
        step (float): grid spacing parameter to compute volume, [Bohr]

    Returns:
        dict: main PCM parameters, including dielectric energy and parameters
            of surface segments

    '''
    # get params
    lot = _lot(mf)
    f_epsilon = mf.with_solvent._intermediates['f_epsilon']
    n_segments = len(mf.with_solvent._intermediates['q'])
    area = float(sum(mf.with_solvent.surface['area'])) # np.float is not json-seriazable
    volume = float(get_sas_volume(mf.with_solvent.surface, step))
    E_tot = float(sum(mf.scf_summary.values()))
    E_diel = float(mf.scf_summary['e_solvent'])

    # atoms
    atom_idxs = list(range(1, 1 + mf.mol.natm))
    a_xs, a_ys, a_zs = mf.mol.atom_coords().T.tolist()
    elems = [elem.lower() for elem in mf.mol.elements]
    radii = [mf.with_solvent.surface['R_vdw'][i] for i, j in mf.with_solvent.surface['gslice_by_atom']]

    # segments
    s_xs, s_ys, s_zs = zip(*mf.with_solvent.surface['grid_coords'].tolist())
    segment_idxs = list(range(1, len(s_xs) + 1))
    atom_segment_idxs = []
    for idx, (start, end) in enumerate(mf.with_solvent.surface['gslice_by_atom']):
        atom_segment_idxs += [idx + 1] * (end - start)
    charges = mf.with_solvent._intermediates['q']
    areas = mf.with_solvent.surface['area']
    potentials = mf.with_solvent._intermediates['v_grids']

    # output
    params = {
        'pyscf_version': __version__,
        'approximation': lot,
        'pcm_data': {
            'f_eps': f_epsilon,
            'n_segments': n_segments,
            'area': float(area),
            'volume': float(volume),
        },
        'screening_charge': {
            'cosmo': float(sum(charges)),
            'correction': 0.0,
            'total': float(sum(charges))
        },
        'energies': {
            'e_tot': float(E_tot),
            'e_tot_corr': float(E_tot), # TODO: implement OCC
            'e_diel': float(E_diel),
            'e_diel_corr': float(E_diel) # TODO: implement OCC
        },
        'atoms': {
            'atom_index': atom_idxs,
            'x': a_xs,
            'y': a_ys,
            'z': a_zs,
            'element': elems,
            'radius': radii
        },
        'segments': {
            'segment_index': segment_idxs,
            'atom_index': atom_segment_idxs,
            'x': s_xs,
            'y': s_ys,
            'z': s_zs,
            'charge': charges.tolist(),
            'charge_corr': charges.tolist(), # TODO: implement OCC
            'area': areas.tolist(),
            'sigma': (charges / areas).tolist(),
            'sigma_corr': (charges / areas).tolist(), # TODO: implement OCC
            'potential': potentials.tolist(),
        }
    }

    return params



def write_cosmo_file(fout, mf, step=0.2, volume=None):
    '''Saves COSMO file in Turbomole format. Please note, that outlying charge
    correction is not implemented yet

    Arguments:
        fout: writable file object
        mf: processed SCF with PCM solvation
        step (float): grid spacing parameter to compute volume, [Bohr]
        volume (float): uses given value (Bohr^3) as molecular volume
            instead of the computed one

    Raises:
        ValueError: if f_epsilon < 1

    '''
    # extract PCM parameters
    ps = get_pcm_parameters(mf, step)

    # check F_epsilon
    f_eps = ps['pcm_data']['f_eps']
    if f_eps < 1.0:
        message  = f'Low f_eps value: {f_eps}. '
        message += 'COSMO-RS requires f_epsilon=1.0 or eps=float("inf"). '
        message += 'Rerun computation with the correct epsilon value.'
        raise ValueError(message)

    # qm params
    fout.write('$info\n')
    info = [
        f'program: PySCF, version {ps["pyscf_version"]}',
        f'solvation model: {ps["approximation"]["solvation_model"]}',
        f'dispersion: {ps["approximation"]["dispersion"]}',
        f'{ps["approximation"]["method"]}',
        f'{ps["approximation"]["basis"]}'
    ]
    fout.write(';'.join(info) + '\n')

    # cosmo data
    n_segments = ps['pcm_data']['n_segments']
    area = ps['pcm_data']['area']
    if volume is None:
        volume = ps['pcm_data']['volume']
    # print
    fout.write('$cosmo_data\n')
    fout.write(f'  fepsi  = {f_eps:>.8f}\n')
    fout.write(f'  nps    = {n_segments:>10d}\n')
    fout.write(f'  area   = {area:>10.2f}       # [a.u.]\n')
    fout.write(f'  volume = {volume:>10.2f}       # [a.u.]\n')

    # atomic coordinates
    atoms = ps['atoms']['atom_index']
    xs, ys, zs = (ps['atoms'][ax] for ax in 'xyz')
    elems = ps['atoms']['element']
    radii = [r * _BOHR for r in ps['atoms']['radius']]
    # print
    fout.write('$coord_rad\n')
    fout.write('#atom   x [a.u.]           y [a.u.]           z [a.u.]      element  radius [A]\n')
    for atom, x, y, z, elem, r in zip(atoms, xs, ys, zs, elems, radii):
        fout.write(f'{atom:>4d}{x:>19.14f}{y:>19.14f}{z:>19.14f}  {elem:<2}{r:>12.5f}\n')

    # screening charges
    charge_cosmo = ps['screening_charge']['cosmo']
    charge_correction = ps['screening_charge']['correction']
    charge_total = ps['screening_charge']['total']
    # print
    fout.write('$screening_charge\n')
    fout.write(f'  cosmo      = {charge_cosmo:>10.6f}\n')
    fout.write(f'  correction = {charge_correction:>10.6f}\n')
    fout.write(f'  total      = {charge_total:>10.6f}\n')

    # energies
    E_tot = ps['energies']['e_tot']
    E_tot_corr = ps['energies']['e_tot_corr']
    E_diel = ps['energies']['e_diel']
    E_diel_corr = ps['energies']['e_diel_corr']
    fout.write('$cosmo_energy\n')
    fout.write(f'  Total energy [a.u.]            = {E_tot:>19.10f}\n')
    fout.write(f'  Total energy + OC corr. [a.u.] = {E_tot_corr:>19.10f}\n')
    fout.write(f'  Total energy corrected [a.u.]  = {E_tot_corr:>19.10f}\n')
    fout.write(f'  Dielectric energy [a.u.]       = {E_diel:>19.10f}\n')
    fout.write(f'  Diel. energy + OC corr. [a.u.] = {E_diel_corr:>19.10f}\n')

    # segments
    ns = ps['segments']['segment_index']
    atoms = ps['segments']['atom_index']
    xs, ys, zs = (ps['segments'][ax] for ax in 'xyz')
    qs = ps['segments']['charge_corr']
    areas = [a * _BOHR**2 for a in ps['segments']['area']]
    pots = [p / _BOHR for p in ps['segments']['potential']]
    # print legend
    fout.write('$segment_information\n')
    fout.write('# n             - segment number\n')
    fout.write('# atom          - atom associated with segment n\n')
    fout.write('# position      - segment coordinates, [a.u.]\n')
    fout.write('# charge        - segment charge (corrected)\n')
    fout.write('# area          - segment area [A**2]\n')
    fout.write('# potential     - solute potential on segment (A length scale)\n')
    fout.write('#\n')
    fout.write('#  n   atom              position (X, Y, Z)                   ')
    fout.write('charge         area        charge/area     potential\n')
    fout.write('#\n')
    fout.write('#\n')
    # print params
    for n, x, y, z, atom, q, area, pot in zip(ns, xs, ys, zs, atoms, qs, areas, pots):
        line  = f'{n:>5d}{atom:>5d}{x:>15.9f}{y:>15.9f}{z:>15.9f}'
        line += f'{q:>15.9f}{area:>15.9f}{q/area:>15.9f}{pot:>15.9f}\n'
        fout.write(line)

    return


