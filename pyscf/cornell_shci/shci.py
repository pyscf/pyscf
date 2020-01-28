#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun
#         Junhao Li
#

'''
SHCI solver for CASCI and CASSCF.

Cornell SHCI program Arrow is developed by Cyrus Umrigar, Junhao Li.  You'll
need to contact Cyrus Umrigar to get the program.
'''

import os
import sys
import json
import time
import tempfile
import copy
import glob
import shutil
from subprocess import check_call, check_output, CalledProcessError
import numpy
from pyscf.lib import logger
from pyscf import lib
from pyscf import tools
from pyscf import ao2mo
from pyscf import mcscf
from pyscf.cornell_shci import symmetry

# Settings
try:
   from pyscf.cornell_shci import settings
except ImportError:
    from pyscf import __config__
    settings = lambda: None
    settings.SHCIEXE = getattr(__config__, 'shci_SHCIEXE', None)
    settings.SHCIRUNTIMEDIR = getattr(__config__, 'shci_SHCIRUNTIMEDIR', None)
    settings.MPIPREFIX = getattr(__config__, 'shci_MPIPREFIX', None)
    if settings.SHCIEXE is None:
        import sys
        sys.stderr.write('settings.py not found for module cornell_shci.  Please create %s\n'
                         % os.path.join(os.path.dirname(__file__), 'settings.py'))
        raise ImportError('settings.py not found')

try:
    sys.path.append(os.path.dirname(settings.SHCIEXE))
    from hc_client import HcClient
except:
    pass

# The default parameters in config file
CONFIG = {
    'system': 'chem',
# Define the number of electrons and spin
    'n_up': 0,
    'n_dn': 0,
    'eps_vars': [
        5e-5,
#        2e-5,
#        1e-5
    ],
    'eps_vars_schedule': [
        2e-3,
        1e-3,
        5e-4,
        2e-4,
        1e-4
    ],
    'chem': {
        # d2h and its subgroups, and Dooh
        'point_group': 'C1'
    },

## Error tol of PT energy. The variational energy error tol equals
## target_error/5000
#    'target_error': 1e-5,
#
## Whether to compute density matrices
#    'get_1rdm_csv': False,
#    'get_2rdm_csv': False,
#
## Variational calculation only, without perturbation correction
#    'var_only' : False,
#
## Set it for Green's function G+
#    'get_green' : False,
#    'w_green' : -0.40,  # frequency
#    'n_green' : 0.01,   # imaginary part to avoid divergence
#
## set it for G-
#     'adavanced' : False,
}

def cleanup(shciobj, remove_wf=False):
    files = ['1rdm.csv',
             '2rdm.csv',
             shciobj.configfile,
             shciobj.integralfile,
             shciobj.outputfile,
             'integrals_cache.dat',
             'result.json',
             ]
    if remove_wf:
        wfn_files = glob.glob(os.path.join(shciobj.runtimedir, 'wf_*'))
        for f in wfn_files:
            os.remove(f)
    
    for f in files:
        if os.path.isfile(os.path.join(shciobj.runtimedir, f)):
            os.remove(os.path.join(shciobj.runtimedir, f))

class SHCI(lib.StreamObject):
    r'''SHCI program interface and object to hold SHCI program input
    parameters.

    See also the homepage of the SHCI program.
    https://github.com/jl2922/shci

    Attributes:

    Examples:

    '''
    def __init__(self, mol=None, tol=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose

        self.executable = settings.SHCIEXE
        self.mpiprefix = settings.MPIPREFIX
        self.runtimedir = '.'#getattr(settings, 'SHCIRUNTIMEDIR', '.')

        self.configfile = 'config.json' # DO NOT modify
        self.integralfile = 'FCIDUMP'   # DO NOT modify
        self.outputfile = 'output.dat'
        self.nroots = 1
        self.conv_tol = tol

        self.config = copy.deepcopy(CONFIG)

        # TODO: Organize into pyscf and SHCI parameters
        self.restart = False
        self.spin = None
        if mol is not None and mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None
        self.dryrun = False

        ##################################################
        #DO NOT CHANGE these parameters, unless you know the code in details
        self.orbsym = []
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** SHCI flags ********')
        log.info('executable    = %s', self.executable)
        log.info('mpiprefix     = %s', self.mpiprefix)
        log.info('runtimedir    = %s', self.runtimedir)
        log.debug1('config = %s', self.config)
        log.info('')
        return self

    def make_rdm1(self, state, norb, nelec, **kwargs):
        dm_file = os.path.join(self.runtimedir, '1rdm.csv')
        if not ('get_1rdm_csv' in self.config and
                os.path.isfile(dm_file) and
                os.path.isfile(get_wfn_file(self, state))):
            write_config(self, nelec, {'get_1rdm_csv': True,
                                       'load_integrals_cache': True})
            execute_shci(self)

        i, j, val = numpy.loadtxt(dm_file, dtype=numpy.dtype('i,i,d'),
                                  delimiter=',', skiprows=1, unpack=True)
        rdm1 = numpy.zeros((norb,norb))
        rdm1[i,j] = rdm1[j,i] = val
        return rdm1

    def make_rdm1s(self, state, norb, nelec, **kwargs):
        # Ref: IJQC, 109, 3552 Eq (3)
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = (nelec-self.spin) // 2
            neleca = nelec - nelecb
        else :
            neleca, nelecb = nelec
        dm1, dm2 = self.make_rdm12(state, norb, nelec, **kwargs)
        dm1n = (2-(neleca+nelecb)/2.) * dm1 - numpy.einsum('pkkq->pq', dm2)
        dm1n *= 1./(neleca-nelecb+1)
        dm1a, dm1b = (dm1+dm1n)*.5, (dm1-dm1n)*.5
        return dm1a, dm1b

    def make_rdm12(self, state, norb, nelec, **kwargs):
        dm_file = os.path.join(self.runtimedir, '2rdm.csv')
        if not ('get_2rdm_csv' in self.config and
                os.path.isfile(dm_file) and
                os.path.isfile(get_wfn_file(self, state))):
            write_config(self, nelec, {'get_2rdm_csv': True,
                                       'load_integrals_cache': True})
            execute_shci(self)

        # two_rdm is dumped as
        # for (unsigned p = 0; p < n_orbs; p++)
        #   for (unsigned q = p; q < n_orbs; q++)
        #     for (unsigned s = 0; s < n_orbs; s++)
        #       for (unsigned r = 0; r < n_orbs; r++) {
        #         if (p == q && s > r) continue;
        #         const double rdm_pqrs = two_rdm[combine4_2rdm(p, q, r, s, n_orbs)];
        #         if (std::abs(rdm_pqrs) < 1.0e-9) continue;
        #         fprintf(pFile, "%d,%d,%d,%d,%#.15g\n", p, q, r, s, rdm_pqrs); }
        i, j, k, l, val = numpy.loadtxt(dm_file, dtype=numpy.dtype('i,i,i,i,d'),
                                        delimiter=',', skiprows=1, unpack=True)
        rdm2 = numpy.zeros((norb,norb,norb,norb))
        rdm2[i,j,k,l] = rdm2[j,i,l,k] = val
        # convert rdm2 to the pyscf convention
        rdm2 = rdm2.transpose(0,3,1,2)

        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]
        rdm1 = numpy.einsum('ikjj->ki', rdm2) / (nelectrons - 1)
        return rdm1, rdm2

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, restart=None,
               **kwargs):
        if restart is None:
            restart = self.restart
        state_id = min(self.config['eps_vars'])
        
        if restart or ci0 is not None:
            if self.verbose >= logger.DEBUG1:
                logger.debug1(self, 'restart was set. wf is read from wf_eps* file.')
            self.cleanup(remove_wf=False)
            wfn_file = get_wfn_file(self, state_id)
            if os.path.isfile(wfn_file):
                shutil.move(wfn_file, get_wfn_file(self, state_id * 2))
        else:
            self.cleanup(remove_wf=True)

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        conf = {}
        if 'tol' in kwargs:
            conf['tol'] = kwargs['tol']
        write_config(self, nelec, conf)

        if self.dryrun:
            logger.info(self, 'Only write integrals and config')
            if self.nroots == 1:
                calc_e = 0.0
                roots = ''
            else :
                calc_e = [0.0] * self.nroots
                roots = [''] * self.nroots
            return calc_e, roots

        if self.nroots != 1:
            raise NotImplementedError

        execute_shci(self)
        if self.verbose >= logger.DEBUG1:
            with open(os.path.join(self.runtimedir, self.outputfile), 'r') as f:
                self.stdout.write(f.read())

        calc_e = read_energy(self)

        # Each eps_vars is associated to one approximate wfn.
        roots = state_id = min(self.config['eps_vars'])
        if not os.path.isfile(get_wfn_file(self, state_id)):
            raise RuntimeError('Eigenstate %s not found' % get_wfn_file(self, state_id))
        return calc_e, roots

    def approx_kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0,
                      restart=None, **kwargs):
        if restart is None:
            restart = self.restart
        state_id = min(self.config['eps_vars'])

        if restart or ci0 is not None:
            self.cleanup(remove_wf=False)
            wfn_file = get_wfn_file(self, state_id)
            if os.path.isfile(wfn_file):
                shutil.move(wfn_file, get_wfn_file(self, state_id * 2))
        else:
            self.cleanup(remove_wf=True)

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        # approx_kernel is called by CASSCF solver only. 2pdm is always needed.
        conf = {'get_2rdm_csv': True}
        if 'tol' in kwargs:
            conf['tol'] = kwargs['tol']
        else:
            conf['tol'] = self.conv_tol * 1e3
        write_config(self, nelec, conf)

        execute_shci(self)
        if self.verbose >= logger.DEBUG1:
            with open(os.path.join(self.runtimedir, self.outputfile), 'r') as f:
                self.stdout.write(f.read())

        calc_e = read_energy(self)

        # Each eps_vars is associated to one approximate wfn.
        roots = state_id = min(self.config['eps_vars'])
        if not os.path.isfile(get_wfn_file(self, state_id)):
            raise RuntimeError('Eigenstate %s not found' % get_wfn_file(self, state_id))
        return calc_e, roots

    def spin_square(self, civec, norb, nelec):
        state_id = civec
        if not ('s2' in self.config and
                os.path.isfile(get_wfn_file(self, state_id))):
            write_config(self, nelec, {'s2': True,
                                       'load_integrals_cache': True})
            execute_shci(self)

        result = get_result(self)
        ss = result['s2']
        s = numpy.sqrt(ss+.25) - .5
        return ss, s*2+1

    def contract_2e(self, eri, civec, norb, nelec, client=None, **kwargs):
        if client is None:
            if getattr(self, '_client', None):
                if not (os.path.isfile(os.path.join(self.runtimedir, self.integralfile)) and
                        os.path.isfile(os.path.join(self.runtimedir, self.configfile))):
                    raise RuntimeError('FCIDUMP or config.json not found')

                self._client = HcClient(nProcs=1, shciPath=self.executable,
                                        runtimePath=self.runtimedir)
                client.startServer()
            client = self._client
        else:
            self._client = client

        return client.Hc(civec)

    cleanup = cleanup

class NpEncoder(json.JSONEncoder):
    """
    Used for dump numpy objects in python3.
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
    
def write_config(shciobj, nelec, config):
    conf = shciobj.config.copy()

    if isinstance(nelec, (int, numpy.integer)):
        if shciobj.spin is None:
            nelecb = nelec // 2
        else:
            nelecb = (nelec - shciobj.spin) // 2
        neleca = nelec - nelecb
    else :
        neleca, nelecb = nelec
    conf['n_up'] = neleca
    conf['n_dn'] = nelecb

    if shciobj.groupname is not None:
        conf['chem']['point_group'] = shciobj.groupname

    if shciobj.conv_tol is not None:
        conf['target_error'] = shciobj.conv_tol * 5000

    conf.update(config)

    if config.get('tol', None) is not None:
        conf['target_error'] = config['tol'] * 5000
    
    with open(os.path.join(shciobj.runtimedir, shciobj.configfile), 'w') as f:
        json.dump(conf, f, indent=2, cls=NpEncoder)

def writeIntegralFile(shciobj, h1eff, eri_cas, ncas, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        if shciobj.spin is None:
            nelecb = nelec // 2
        else:
            nelecb = (nelec - shciobj.spin) // 2
        neleca = nelec - nelecb
    else :
        neleca, nelecb = nelec

    if shciobj.groupname is not None and shciobj.orbsym is not []:
# First removing the symmetry forbidden integrals. This has been done using
# the pyscf internal irrep-IDs (stored in shciobj.orbsym)
        orbsym = numpy.asarray(shciobj.orbsym) % 10
        pair_irrep = (orbsym.reshape(-1,1) ^ orbsym)[numpy.tril_indices(ncas)]
        sym_forbid = pair_irrep.reshape(-1,1) != pair_irrep.ravel()
        eri_cas = ao2mo.restore(4, eri_cas, ncas)
        eri_cas[sym_forbid] = 0
        eri_cas = ao2mo.restore(8, eri_cas, ncas)
        # Convert the pyscf internal irrep-ID to molpro irrep-ID
        orbsym = numpy.asarray(symmetry.convert_orbsym(shciobj.groupname, orbsym))
    else:
        orbsym = []
        eri_cas = ao2mo.restore(8, eri_cas, ncas)

    if not os.path.exists(shciobj.runtimedir):
        os.makedirs(shciobj.runtimedir)

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(shciobj.runtimedir, shciobj.integralfile)
    tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                 neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                 orbsym=orbsym)
    return integralFile

def execute_shci(shciobj):
    output = os.path.join(shciobj.runtimedir, shciobj.outputfile)
    cmd = ' '.join((shciobj.mpiprefix, shciobj.executable))
    try:
        #cmd = ' '.join((shciobj.mpiprefix, shciobj.executable))
        #cmd = "%s > %s 2>&1" % (cmd, output)
        #check_call(cmd, shell=True, cwd=shciobj.runtimedir)
        with open(output, 'w') as f:
            check_call(cmd.split(), cwd=shciobj.runtimedir, stdout=f, stderr=f)
    except CalledProcessError as err:
        logger.error(shciobj, cmd)
        shciobj.stdout.write(check_output(['tail', '-100', output]))
        raise err
    return output


def get_result(shciobj):
    with open(os.path.join(shciobj.runtimedir, 'result.json'), 'r') as f:
        result = json.load(f)
    return result

def read_energy(shciobj, state_id=None):
    result = get_result(shciobj)

    if state_id is None:
        state_id = '%.2e' % (min(shciobj.config['eps_vars']))

    #FIXME: whether to return result['energy_total']['extrapolate']['value']

    if 'energy_total' in result:
        e = result['energy_total'][state_id].values()[0]['value']
    else:
        e = result['energy_var'][state_id]
    return e

def read_state(shciobj, state_id=None):
    wfn_file = get_wfn_file(shciobj, state_id)
    raise NotImplementedError

def get_wfn_file(shciobj, state_id=None):
    if state_id is None:
        state_id = min(shciobj.config['eps_vars'])
    wfn_file = os.path.join(shciobj.runtimedir, 'wf_eps1_%.2e.dat' % state_id)
    return wfn_file


def SHCISCF(mf, norb, nelec, tol=1.e-8, *args, **kwargs):
    '''Shortcut function to setup CASSCF using the SHCI solver.  The SHCI
    solver is properly initialized in this function so that the 1-step
    algorithm can be applied with SHCI-CASSCF.

    Examples:

    >>> from pyscf.cornell_shci import shci
    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = shci.SHCISCF(mf, 4, 4)
    >>> mc.kernel()
    '''
    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = SHCI(mf.mol, tol=tol)
    mc.fcisolver.config['get_1rdm_csv'] = True
    mc.fcisolver.config['get_2rdm_csv'] = True
    mc.fcisolver.config['var_only'] = True
    mc.fcisolver.config['s2'] = True
    return mc

def dryrun(mc, mo_coeff=None):
    '''Generate FCIDUMP and SHCI config file'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    with lib.temporary_env(mc.fcisolver, dryrun=True):
        mc.casci(mo_coeff)


class shci_client(object):
    '''Run SHCI in client mode for matrix-vector operation H*c

    Examples:

    >>> from pyscf.cornell_shci import shci
    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 8, 8)
    >>> mc.fcisolver = shci.SHCI(mol)
    >>> mc.kernel()
    >>> with shci.shci_client(mc.fcisolver) as sc:
    ...     civec = sc.getCoefs()
    ...     hc = sc.Hc(civec)
    '''
    def __init__(self, shciobj):
        client = HcClient(nProcs=1, shciPath=shciobj.executable,
                          runtimePath=shciobj.runtimedir)
        shciobj._client = client
        self._shciobj = shciobj

    def __enter__(self):
        shciobj = self._shciobj
        if not (os.path.isfile(os.path.join(shciobj.runtimedir, shciobj.integralfile)) and
                os.path.isfile(os.path.join(shciobj.runtimedir, shciobj.configfile))):
            raise RuntimeError('FCIDUMP or config.json not found')

        shciobj._client.startServer()
        return shciobj._client

    def __exit__(self, type, value, traceback):
        self.shciobj._client.exit()


if __name__ == '__main__':
    from pyscf import gto, scf, mcscf
    from pyscf.cornell_shci import shci

    # Initialize N2 molecule
    b = 1.098
    mol = gto.Mole()
    mol.build(
        verbose = 4,
        output = None,
        atom = [
            ['N',(  0.000000,  0.000000, -b/2)],
            ['N',(  0.000000,  0.000000,  b/2)], ],
        basis = {'N': 'ccpvdz', },
        #symmetry=True,
    )

    # Create HF molecule
    mf = scf.RHF( mol ).run()

    # Number of orbital and electrons
    norb = 8
    nelec = 10
    dimer_atom = 'N'

#    mch = mcscf.CASCI(mf, norb, nelec)
#    mch.fcisolver = SHCI(mf.mol)
#    mch.kernel()
#    dm2 = mch.fcisolver.make_rdm12(0, norb, nelec)[1]
#
#    mc1 = mcscf.CASCI(mf, norb, nelec)
#    mc1.kernel(mch.mo_coeff)
#    dm2ref = mc1.fcisolver.make_rdm12(mc1.ci, norb, nelec)[1]
#    print abs(dm2ref-dm2).max()
#    exit()

    mch = shci.SHCISCF( mf, norb, nelec )
    mch.internal_rotation = True
    mch.kernel()

#    mc1 = mcscf.CASSCF(mf, norb, nelec)
#    mc1.kernel()

#    with shci_client(mch.fcisolver) as client:
#        coefs = client.getCoefs()
#        c = client.Hc(coefs)
