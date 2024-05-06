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
Hessian of SMD solvent model, copied from GPU4PySCF with modification for CPU
'''
# pylint: disable=C0103

import numpy as np
from pyscf import scf, lib
from pyscf.solvent import smd
from pyscf.solvent.grad import smd as smd_grad
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.solvent.hessian import pcm as pcm_hess
from pyscf.lib import logger

def hess_elec(smdobj, dm, verbose=None):
    '''
    slow version with finite difference
    TODO: use analytical hess_nuc
    '''
    log = logger.new_logger(smdobj, verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    pmol = smdobj.mol.copy()
    mol = pmol.copy()
    coords = mol.atom_coords(unit='Bohr')

    def smd_grad_scanner(mol):
        # TODO: use more analytical forms
        smdobj.reset(mol)
        smdobj._get_vind(dm)
        grad = pcm_grad.grad_nuc(smdobj, dm)
        grad+= smd_grad.grad_solver(smdobj, dm)
        grad+= pcm_grad.grad_qv(smdobj, dm)
        return grad

    mol.verbose = 0
    de = np.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-3
    for ia in range(mol.natm):
        for ix in range(3):
            dv = np.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            g0 = smd_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            g1 = smd_grad_scanner(mol)
            de[ia,:,ix] = (g0 - g1)/2.0/eps
    t1 = log.timer_debug1('solvent energy', *t1)
    smdobj.reset(pmol)
    return de

def get_cds(smdobj):
    mol = smdobj.mol
    solvent = smdobj.solvent
    def smd_grad_scanner(mol):
        smdobj_tmp = smd.SMD(mol)
        smdobj_tmp.solvent = solvent
        return smd_grad.get_cds(smdobj_tmp)

    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    log.warn("Using finite difference scheme for CDS contribution.")
    eps = 1e-4
    natm = mol.natm
    hess_cds = np.zeros([natm,natm,3,3])
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad0_cds = smd_grad_scanner(mol)

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()
            grad1_cds = smd_grad_scanner(mol)

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            hess_cds[ia,:,j] = (grad0_cds - grad1_cds) / (2.0 * eps)
    t1 = log.timer_debug1('solvent energy', *t1)
    return hess_cds # hartree

def make_hess_object(hess_method):
    '''For hess_method in vacuum, add nuclear Hessian of solvent smdobj'''
    if hess_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    name = (hess_method.base.with_solvent.__class__.__name__
            + hess_method.__class__.__name__)
    return lib.set_class(WithSolventHess(hess_method),
                         (WithSolventHess, hess_method.__class__), name)

class WithSolventHess:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_gpu(self):
        from gpu4pyscf.solvent.hessian import smd     # type: ignore
        hess_method = self.undo_solvent().to_gpu()
        return smd.make_hess_object(hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        is_equilibrium = self.base.with_solvent.equilibrium_solvation
        self.base.with_solvent.equilibrium_solvation = True
        self.de_solvent = hess_elec(self.base.with_solvent, dm, verbose=self.verbose)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de_cds = get_cds(self.base.with_solvent)
        self.de = self.de_solute + self.de_solvent + self.de_cds
        self.base.with_solvent.equilibrium_solvation = is_equilibrium
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1(ao_repr=True)
            dv = pcm_hess.fd_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1(ao_repr=True)
            dm = dm[0] + dm[1]
            dva = pcm_hess.fd_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = pcm_hess.fd_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')
    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
