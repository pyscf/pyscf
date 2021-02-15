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

from __future__ import division
import os
import subprocess

def get_siesta_command(label, directory=''):
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

def runbash(cmd):
    """
    Run a bash command with subprocess, fails if ret != 0
    """
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise ValueError(cmd + ": failed with the error {0}".format(ret))


def install_siesta():
    from os.path import expanduser
    home = expanduser("~")
    os.chdir(home)
    siesta_version = "4.1-b2"
    subprocess.call("rm -r siesta-" + siesta_version + "*", shell=True)
    runbash("wget https://launchpad.net/siesta/4.1/4.1-b2/+download/siesta-"\
            + siesta_version + ".tar.gz")
    runbash("tar -xvzf siesta-" + siesta_version + ".tar.gz")
    os.chdir("siesta-" + siesta_version + "/Obj/")
    runbash("sh ../Src/obj_setup.sh")
    runbash("cp gfortran.make arch.make")
    runbash("make -j")

    cwd = os.getcwd()

    siesta_dir = cwd
    pseudo_path = cwd + "/pseudo"
    os.mkdir(pseudo_path)
    os.chdir(pseudo_path)

    species = ['H', 'O', 'Na']
    # seems to not work anymore ??
    # pseudo_src = https://departments.icmab.es/leem/siesta/Databases/Pseudopotentials/Pseudos_LDA_Abinit/" + sp +"_html/"
    pseudo_src = "https://mbarbry.pagekite.me/pseudo/"
    for sp in species:
        runbash("wget " + pseudo_src + sp +".psf")

    os.chdir(home)
    f = open(home + "/siesta_pseudo_path.txt", "w")
    f.write(pseudo_path)
    f.close()

    f = open(home + "/siesta_dir.txt", "w")
    f.write(siesta_dir)
    f.close()

    f = open(home + "/siesta_command.txt", "w")
    f.write("'siesta < %s |tee %s'\n")
    f.close()
