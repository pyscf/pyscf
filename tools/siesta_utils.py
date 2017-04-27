from __future__ import division
import os

def get_siesta_command(label, directory='./'):
    # Setup the siesta command.
    command = os.environ.get('SIESTA_COMMAND')
    if command is None:
        mess = "The 'SIESTA_COMMAND' environment is not defined."
        raise ValueError(mess)

    runfile = directory + label + '.fdf'
    outfile = label + '.out'
    
    try:
        command = command % (runfile, outfile)
        return command
    except TypeError:
        raise ValueError(
            "The 'SIESTA_COMMAND' environment must " +
            "be a format string" +
            " with two string arguments.\n" +
            "Example : 'siesta < ./%s > ./%s'.\n" +
            "Got '%s'" % command)

def get_pseudo(sp, suffix=''):
    """
        return the path to the pseudopotential of a particular specie
    """
    pseudo_path = os.environ['SIESTA_PP_PATH']
    if pseudo_path is None:
        raise ValueError('The SIESTA_PP_PATH environement is not defined.')
    fname = pseudo_path + '/' + sp+suffix + '.psf'

    if os.path.isfile(fname):
        return fname
    else:
        raise ValueError('pseudopotential ' + fname + ' does not exist.')
