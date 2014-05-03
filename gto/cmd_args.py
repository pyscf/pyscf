#
# File: cmd_args.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import optparse
import lib.logger as log

def cmd_args():
    '''
    get input from cmdline
    '''
    parser = optparse.OptionParser()
    parser.add_option('-v', '--verbose',
                      action='store_false', dest='verbose',
                      help='make lots of noise')
    parser.add_option('-q', '--quiet',
                      action='store_false', dest='quite', default=False,
                      help='be very quiet')
    parser.add_option('-o', '--output',
                      dest='output', metavar='FILE', help='write output to FILE')
    parser.add_option('-m', '--max-memory',
                      action='store', dest='max_memory', metavar='NUM',
                      help='maximum memory to use (in MB)')

    (opts, args_left) = parser.parse_args()

    if opts.quite:
        opts.verbose = log.QUITE

    if opts.verbose:
        opts.verbose = log.DEBUG

    if opts.output:
        if os.path.isfile(opts.output):
            if opts.verbose and opts.verbose > log.QUITE:
                os.remove(opts.output)
                print 'overwrite output file: %s' % opts.output
        else:
            if opts.verbose and opts.verbose > log.QUITE:
                print 'output file: %s' % opts.output

    if opts.max_memory:
        opts.max_memory = float(opts.max_memory)

    return opts

if __name__ == '__main__':
    opts = cmd_args()
    print opts.verbose, opts.output, opts.max_memory
