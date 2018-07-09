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

class color:
  import os
  T = os.getenv('TERM')
  if ( T=='cygwin' or T=='mingw' ) :
    HEADER = '\033[01;35m'
    BLUE = '\033[01;34m'
    GREEN = '\033[01;32m'
    WARNING = '\033[01;33m'
    FAIL = '\033[01;31m'
    RED = FAIL
    ENDC = '\033[0m'
  else :
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RED = FAIL
    ENDC = '\033[0m'

  def disable(self):
    self.HEADER = ''
    self.OKBLUE = ''
    self.OKGREEN = ''
    self.WARNING = ''
    self.FAIL = ''
    self.RED = ''
    self.ENDC = ''
