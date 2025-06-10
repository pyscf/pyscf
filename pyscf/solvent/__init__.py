# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

from pyscf.solvent import ddcosmo
from pyscf.solvent import pcm
from pyscf.solvent import smd

def ddCOSMO(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize ddCOSMO model.

    Examples:

    >>> mf = ddCOSMO(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = ddCOSMO(mol)
    >>> mc = ddCOSMO(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from pyscf import scf, mcscf
    from pyscf import tdscf

    if isinstance(method_or_mol, gto.mole.Mole):
        return ddcosmo.DDCOSMO(method_or_mol)

    method = method_or_mol
    if isinstance(method, scf.hf.SCF):
        return ddcosmo.ddcosmo_for_scf(method, solvent_obj, dm)
    elif isinstance(method, mcscf.casci.CASBase):
        if isinstance(method, mcscf.mc1step.CASSCF):
            return ddcosmo.ddcosmo_for_casscf(method, solvent_obj, dm)
        elif isinstance(method, mcscf.casci.CASCI):
            return ddcosmo.ddcosmo_for_casci(method, solvent_obj, dm)
    elif isinstance(method, tdscf.rhf.TDBase):
        return ddcosmo.ddcosmo_for_tdscf(method, solvent_obj, dm)
    elif hasattr(method, '_scf'):
        return ddcosmo.ddcosmo_for_post_scf(method, solvent_obj, dm)
    raise RuntimeError(f'ddCOSMO for {method} not available')
DDCOSMO = ddCOSMO

def ddPCM(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize ddPCM model.

    Examples:

    >>> mf = ddPCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = ddPCM(mol)
    >>> mc = ddPCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from pyscf import scf, mcscf
    from pyscf import tdscf
    from pyscf.solvent import ddpcm

    if isinstance(method_or_mol, gto.mole.Mole):
        return ddpcm.DDPCM(method_or_mol)

    method = method_or_mol
    if isinstance(method, scf.hf.SCF):
        return ddpcm.ddpcm_for_scf(method, solvent_obj, dm)
    elif isinstance(method, mcscf.casci.CASBase):
        if isinstance(method, mcscf.mc1step.CASSCF):
            return ddpcm.ddpcm_for_casscf(method, solvent_obj, dm)
        elif isinstance(method, mcscf.casci.CASCI):
            return ddpcm.ddpcm_for_casci(method, solvent_obj, dm)
    elif isinstance(method, tdscf.rhf.TDBase):
        return ddpcm.ddpcm_for_tdscf(method, solvent_obj, dm)
    elif hasattr(method, '_scf'):
        return ddpcm.ddpcm_for_post_scf(method, solvent_obj, dm)
    raise RuntimeError(f'ddPCM for {method} not available')
DDPCM = ddPCM

def PE(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize polarizable embedding model.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            If method_or_mol is gto.Mole object, this function returns a
            PolEmbed object constructed with this Mole object.
        solvent_obj (PolEmbed object or dictionary with options or str)
            If solvent_obj is an object of PolEmbed class, the PE-enabled
            method will be created using solvent_obj.
            If solvent_obj is dict or str, a PolEmbed object will
            be created first with the solvent_obj, on top of which PE-enabled
            method will be created.

    Examples:

    >>> pe_options = {"potfile": "pyframe.pot"}
    >>> mf = PE(scf.RHF(mol), pe_options)
    >>> mf.kernel()
    '''
    from pyscf.solvent import pol_embed
    from pyscf import gto, scf, mcscf, tdscf

    if isinstance(method_or_mol, gto.mole.Mole):
        return pol_embed.PolEmbed(method_or_mol, solvent_obj)

    method = method_or_mol
    if isinstance(method, scf.hf.SCF):
        assert solvent_obj is not None
        return pol_embed.pe_for_scf(method, solvent_obj, dm)

    if solvent_obj is None:
        solvent_obj = method._scf.with_solvent
    if isinstance(method, mcscf.casci.CASBase):
        if isinstance(method, mcscf.mc1step.CASSCF):
            return pol_embed.pe_for_casscf(method, solvent_obj, dm)
        elif isinstance(method, mcscf.casci.CASCI):
            return pol_embed.pe_for_casci(method, solvent_obj, dm)
    elif isinstance(method, tdscf.rhf.TDBase):
        return pol_embed.pe_for_tdscf(method, solvent_obj, dm)
    elif hasattr(method, '_scf'):
        return pol_embed.pe_for_post_scf(method, solvent_obj, dm)
    raise RuntimeError(f'PolEmbed for {method} not available')

def PCM(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize PCM model.

    Examples:

    >>> mf = PCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = PCM(mol)
    >>> mc = PCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from pyscf import scf, mcscf
    from pyscf import tdscf

    if isinstance(method_or_mol, gto.mole.Mole):
        return pcm.PCM(method_or_mol)

    method = method_or_mol
    if isinstance(method, scf.hf.SCF):
        return pcm.pcm_for_scf(method, solvent_obj, dm)
    elif isinstance(method, mcscf.casci.CASBase):
        if isinstance(method, mcscf.mc1step.CASSCF):
            return pcm.pcm_for_casscf(method, solvent_obj, dm)
        elif isinstance(method, mcscf.casci.CASCI):
            return pcm.pcm_for_casci(method, solvent_obj, dm)
    elif isinstance(method, tdscf.rhf.TDBase):
        return pcm.pcm_for_tdscf(method, solvent_obj, dm)
    elif hasattr(method, '_scf'):
        return pcm.pcm_for_post_scf(method, solvent_obj, dm)
    raise RuntimeError(f'PCM for {method} not available')

PCM = PCM

def SMD(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize PCM model.

    Examples:

    >>> mf = PCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = PCM(mol)
    >>> mc = PCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from pyscf import scf

    method = method_or_mol
    if isinstance(method, gto.mole.Mole):
        return smd.SMD(method)
    elif isinstance(method, scf.hf.SCF):
        return smd.smd_for_scf(method, solvent_obj, dm)
    raise RuntimeError(f'SMD for {method} not available')

SMD = SMD
