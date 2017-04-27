from pyscf.tools import fcidump
from pyscf.tools import molden
from pyscf.tools import dump_mat

#
#
#
class DevNull:
    """
        Class use in the testing routines.
        Coming from ase/ase/utils/__init__.py
    """
    encoding = 'UTF-8'
    
    def write(self, string):
        pass
    
    def flush(self):
        pass
    
    def seek(self, offset, whence=0):
        return 0
    
    def tell(self):
        return 0
    
    def close(self):
        pass
    
    def isatty(self):
        return False

devnull = DevNull()
