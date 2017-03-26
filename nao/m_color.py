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
