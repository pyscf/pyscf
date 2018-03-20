#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Tune CASSCF solver on the fly.

However, it is unrecommended to tune CASSCF solver on the runtime unless you
know exactly what you're doing.
'''

from pyscf import gto, scf, mcscf

def hot_tuning_(casscf, configfile=None):
    '''Allow you to tune CASSCF parameters at the runtime
    '''
    import os
    import sys
    import traceback
    import tempfile
    import json
    #from numpy import array
    from pyscf.lib import logger

    if configfile is None:
        fconfig = tempfile.NamedTemporaryFile(suffix='.json')
        configfile = fconfig.name
    logger.info(casscf, 'Function hot_tuning_ dumps CASSCF parameters in config file%s',
                configfile)

    exclude_keys = set(('stdout', 'verbose', 'ci', 'mo_coeff', 'mo_energy',
                        'e_cas', 'e_tot', 'ncore', 'ncas', 'nelecas', 'mol',
                        'callback', 'fcisolver'))

    casscf_settings = {}
    for k, v in casscf.__dict__.items():
        if not (k.startswith('_') or k in exclude_keys):
            if (v is None or
                isinstance(v, (str, bool, int, float, list, tuple, dict))):
                casscf_settings[k] = v
            elif isinstance(v, set):
                casscf_settings[k] = list(v)

    doc = '''# JSON format
# Note the double quote "" around keyword
'''
    conf = {'casscf': casscf_settings}
    with open(configfile, 'w') as f:
        f.write(doc)
        f.write(json.dumps(conf, indent=4, sort_keys=True) + '\n')
        f.write('# Starting from this line, code are parsed as Python script.  The Python code\n'
                '# will be injected to casscf.kernel through callback hook.  The casscf.kernel\n'
                '# function local variables can be directly accessed.  Note, these variables\n'
                '# cannot be directly modified because the environment is generated using\n'
                '# locals() function (see\n'
                '# https://docs.python.org/2/library/functions.html#locals).\n'
                '# You can modify some variables with inplace updating, eg\n'
                '# from pyscf import fci\n'
                '# if imacro > 6:\n'
                '#     casscf.fcislover = fci.fix_spin_(fci.direct_spin1, ss=2)\n'
                '#     mo[:,:3] *= -1\n'
                '# Warning: this runtime modification is unsafe and highly unrecommended.\n')

    old_cb = casscf.callback
    def hot_load(envs):
        try:
            with open(configfile) as f:
# filter out comments
                raw_js = []
                balance = 0
                data = [x for x in f.readlines()
                        if not x.startswith('#') and x.rstrip()]
                for n, line in enumerate(data):
                    if not line.lstrip().startswith('#'):
                        raw_js.append(line)
                        balance += line.count('{') - line.count('}')
                        if balance == 0:
                            break
            raw_py = ''.join(data[n+1:])
            raw_js = ''.join(raw_js)

            logger.debug(casscf, 'Reading CASSCF parameters from config file  %s',
                         os.path.realpath(configfile))
            logger.debug1(casscf, '    Inject casscf settings %s', raw_js)
            conf = json.loads(raw_js)
            casscf.__dict__.update(conf.pop('casscf'))

            # Not yet found a way to update locals() on the runtime
            # https://docs.python.org/2/library/functions.html#locals
            #for k in conf:
            #    if k in envs:
            #        logger.info(casscf, 'Update envs[%s] = %s', k, conf[k])
            #        envs[k] = conf[k]

            logger.debug1(casscf, '    Inject python script\n%s\n', raw_py)
            if len(raw_py.strip()) > 0:
                if sys.version_info >= (3,):
# A hacky call using eval because exec are so different in python2 and python3
                    eval(compile('exec(raw_py, envs, {})', '<str>', 'exec'))
                else:
                    eval(compile('exec raw_py in envs, {}', '<str>', 'exec'))
        except Exception as e:
            logger.warn(casscf, 'CASSCF hot_load error %s', e)
            logger.warn(casscf, ''.join(traceback.format_exc()))

        if callable(old_cb):
            old_cb(envs)

    casscf.callback = hot_load
    return casscf


b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 5,
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
)
mf = scf.RHF(mol)
mf.kernel()

#
# This step creates a hook on CASSCF callback function.  It allows the CASSCF
# solver reading the contents of config and update the solver itself in every
# micro iteration.
#
# Then start the casscf solver.
#
mc = hot_tuning_(mcscf.CASSCF(mf, 6, 6), 'config')
mc.kernel()

#
# The solver finishes quickly for this system since it is small.  Assuming the
# system is large and the optimization iteration processes slowly,  we can
# modify the content of config during the optimization, to change the behavior
# of CASSCF solver.  Eg
#
# 1. We can set the "frozen" attribute in the config file, to avoid the
# orbital rotation over optimized.  After a few optimization cycles, we can
# reset "frozen" to null, to restore the regular CASSCF optimization.
#
# 2. We can tune ah_start_cycle to increase the AH solver accuracy.  Big value
# in ah_start_cycle can postpone the orbital rotation with better AH solution.
# It may help predict a better step in orbital rotation function.
#
