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
Gradient of SMD solvent model, copied from GPU4PySCF with modification for CPU
'''
# pylint: disable=C0103

import numpy as np
from pyscf import lib
from pyscf.grad import rhf as rhf_grad

from pyscf.solvent import pcm, smd
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.lib import logger

def get_cds(smdobj):
    return smd.get_cds_legacy(smdobj)[1]

def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    name = (grad_method.base.with_solvent.__class__.__name__
            + grad_method.__class__.__name__)
    return lib.set_class(WithSolventGrad(grad_method),
                         (WithSolventGrad, grad_method.__class__), name)

class WithSolventGrad:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_gpu(self):
        from gpu4pyscf.solvent.grad import smd      # type: ignore
        grad_method = self.undo_solvent().to_gpu()
        return smd.make_grad_object(grad_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        self.de_solute  = super().kernel(*args, **kwargs)
        self.de_solvent = pcm_grad.grad_qv(self.base.with_solvent, dm)
        self.de_solvent+= pcm_grad.grad_solver(self.base.with_solvent, dm)
        self.de_solvent+= pcm_grad.grad_nuc(self.base.with_solvent, dm)
        #self.de_cds     = get_cds(self.base.with_solvent)
        self.de_cds     = smd.get_cds_legacy(self.base.with_solvent)[1]
        self.de = self.de_solute + self.de_solvent + self.de_cds

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


