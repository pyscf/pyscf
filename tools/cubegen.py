#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.dft import numint, gen_grid

'''
Gaussian cube file format
'''

def density(mol, outfile, dm, nx=80, ny=80, nz=80):
    coord = mol.atom_coords()
    box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6
    boxorig = numpy.min(coord,axis=0) - 3
    xs = numpy.arange(nx) * (box[0]/nx)
    ys = numpy.arange(ny) * (box[1]/ny)
    zs = numpy.arange(nz) * (box[2]/nz)
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = numpy.asarray(coords, order='C') - (-boxorig)

    nao = mol.nao_nr()
    ngrids = nx * ny * nz
    blksize = min(200, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx,ny,nz)

    with open(outfile, 'w') as f:
        f.write('Density in real space\n')
        f.write('Comment line\n')
        f.write('%5d' % mol.natm)
        f.write(' %14.8f %14.8f %14.8f\n' % tuple(boxorig.tolist()))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (nx, xs[1], 0, 0))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (ny, 0, ys[1], 0))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (nz, 0, 0, zs[1]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d %f' % (chg, chg))
            f.write(' %14.8f %14.8f %14.8f\n' % tuple(coord[ia]))
        fmt = ' %14.8e' * nz + '\n'
        for ix in range(nx):
            for iy in range(ny):
                f.write(fmt % tuple(rho[ix,iy].tolist()))


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='H 0 0 0; H 0 0 1')
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2.cube', mf.make_rdm1())

