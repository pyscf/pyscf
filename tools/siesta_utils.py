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
