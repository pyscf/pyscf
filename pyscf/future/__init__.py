import sys
sys.stderr.write('''

Warning

Modules in the "future" directory (dmrgscf, fciqmcscf, shciscf, icmspt, xianci)
have been moved to pyscf/pyscf directory.  You can still import these modules.
from the "future" directory, and they work the same as before.

To avoid name conflicts with python built-in module "future", this directory
will be deleted in future release.

''')

