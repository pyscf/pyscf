#!/bin/bash
#
# This issue has been solved in PR #149.  Remove me in the future
#


# On Mac OSX 10.11 or newer, you may get error message if you use clang compiler
#
#  OSError: dlopen(xxx/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.3.0.dylib
#  Referenced from: xxx/pyscf/lib/libcgto.dylib
#  Reason: unsafe use of relative rpath libcint.3.0.dylib in xxx/pyscf/lib/libao2mo.dylib with restricted binary
#
# It requires following fixing

dirnow=$(pwd)/$(dirname $0)

cd $dirnow
for i in *.dylib
do
  echo install_name_tool -change libcint.3.0.dylib ${dirnow}/deps/lib/libcint.3.0.dylib ${dirnow}/$i
  install_name_tool -change libcint.3.0.dylib ${dirnow}/deps/lib/libcint.3.0.dylib ${dirnow}/$i
  echo install_name_tool -change libxcfun.dylib ${dirnow}/deps/lib/libxcfun.dylib ${dirnow}/$i
  install_name_tool -change libxcfun.dylib ${dirnow}/deps/lib/libxcfun.dylib ${dirnow}/$i
done
