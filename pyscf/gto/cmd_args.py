# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import pyscf.lib.logger

import argparse

def cmd_args():
    '''
    get input from cmdline
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        action='store_false', dest='verbose', default=0,
                        help='make lots of noise')
    parser.add_argument('-q', '--quiet',
                        action='store_false', dest='quite', default=False,
                        help='be very quiet')
    parser.add_argument('-o', '--output',
                        dest='output', metavar='FILE', help='write output to FILE')
    parser.add_argument('-m', '--max-memory',
                        action='store', dest='max_memory', metavar='NUM',
                        help='maximum memory to use (in MB)')

    (opts, args_left) = parser.parse_known_args()

    if opts.quite:
        opts.verbose = pyscf.lib.logger.QUIET

    if opts.verbose:
        opts.verbose = pyscf.lib.logger.DEBUG

    if opts.max_memory:
        opts.max_memory = float(opts.max_memory)

    return opts


if __name__ == '__main__':
    opts = cmd_args()
    print(opts.verbose, opts.output, opts.max_memory)
