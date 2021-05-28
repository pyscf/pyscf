import argparse
import sys

DEFAULT_LOGNAME = 'embcc'
DEFAULT_LOGLVL = 10
# In future, INFO level will be default:
#DEFAULT_LOGLVL = 20

def parse_cmd_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('-o', '--output', dest='logname', default=DEFAULT_LOGNAME)
    parser.add_argument('-v', action='store_const', dest='loglevel', const=15, default=DEFAULT_LOGLVL)
    parser.add_argument('-vv', action='store_const', dest='loglevel', const=10)
    parser.add_argument('-vvv', action='store_const', dest='loglevel', const=1)
    parser.add_argument('--strict', action='store_true')
    args, unknown_args = parser.parse_known_args()

    # Remove known arguments:
    sys.argv = [sys.argv[0], *unknown_args]

    return args
