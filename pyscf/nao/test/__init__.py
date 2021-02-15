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

from __future__ import division, print_function
import os
import sys
import subprocess
import unittest
import platform
import tempfile
import shutil
from importlib import import_module
from glob import glob

from os import devnull

#
#
#
class NotAvailable(Exception):
    pass

#
#
#
class ScriptTestCase(unittest.TestCase):
    def __init__(self, methodname='testfile', filename=None):
        unittest.TestCase.__init__(self, methodname)
        self.filename = filename

    def testfile(self):
        try:
            with open(self.filename) as fd:
                exec(compile(fd.read(), self.filename, 'exec'), {})
        except KeyboardInterrupt:
            raise RuntimeError('Keyboard interrupt')
        except ImportError as ex:
            module = ex.args[0].split()[-1].replace("'", '').split('.')[0]
            if module in ['scipy', 'matplotlib', 'Scientific', 'lxml',
                          'flask', 'argparse']:
                sys.__stdout__.write('skipped (no {0} module) '.format(module))
            else:
                raise
        except NotAvailable as notavailable:
            sys.__stdout__.write('skipped ')
            msg = str(notavailable)
            if msg:
                sys.__stdout__.write('({0}) '.format(msg))

    def id(self):
        return self.filename

    def __str__(self):
        return self.filename.split('test/')[-1]

    def __repr__(self):
        return "ScriptTestCase(filename='%s')" % self.filename

#
#
#
def test(verbosity=1, testdir=None, stream=sys.stdout, files=None, siesta_exe='siesta'):
    """
    files : 
    """

    
    ts = unittest.TestSuite()
    if files:
        files = [os.path.join(__path__[0], f) for f in files]
    else:
        files = glob(__path__[0] + '/*')
    
    sdirtests = []
    tests = []

    # look files in sub dir
    for f in files:
        # look first level sub dir
        if os.path.isdir(f):
            files_sub = glob(f+'/*')
            sdirtests.extend(glob(f + '/*.py'))
            
            # second level sub dir
            for fsub in files_sub:
                if os.path.isdir(fsub):
                    sdirtests.extend(glob(fsub + '/*.py'))
                else:
                    if fsub.endswith('.py'):
                        tests.append(fsub)
        else:
            if f.endswith('.py'):
                tests.append(f)

    for test in tests + sdirtests:
        if test.endswith('__.py'):
            continue
        ts.addTest(ScriptTestCase(filename=os.path.abspath(test)))

    versions = [('platform', platform.platform()),
            ('python-' + sys.version.split()[0], sys.executable)]

    for name in ['pyscf', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            versions.append((name, 'no'))
        else:
            versions.append((name + '-' + module.__version__,
                    module.__file__.rsplit('/', 1)[0] + '/'))

    if verbosity:
        for a, b in versions:
            print('{0:16}{1}'.format(a, b))
    
    sys.stdout = open(devnull, 'w')
    if verbosity == 0:
        stream = open(devnull, 'w')
    ttr = unittest.TextTestRunner(verbosity=verbosity, stream=stream)

    origcwd = os.getcwd()

    if testdir is None:
        testdir = tempfile.mkdtemp(prefix='pyscf-test-')
    else:
        if os.path.isdir(testdir):
            shutil.rmtree(testdir)  # clean before running tests!
        os.mkdir(testdir)
    os.chdir(testdir)

    if verbosity:
        print('test-dir       ', testdir, '\n', file=sys.__stdout__)
    try:
        results = ttr.run(ts)
    finally:
        os.chdir(origcwd)
        sys.stdout = sys.__stdout__

    return results
