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

import sys
sys.stderr.write('''

Warning

Modules in the "future" directory (dmrgscf, fciqmcscf, shciscf, icmspt, xianci)
have been moved to pyscf/pyscf directory.  You can still import these modules.
from the "future" directory, and they work the same as before.

To avoid name conflicts with python built-in module "future", this directory
will be deleted in future release.

''')

