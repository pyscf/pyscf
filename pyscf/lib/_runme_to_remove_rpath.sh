#!/bin/bash
# On Mac OSX 10.11 or newer, you may get error message if you use clang compiler
#
#  OSError: dlopen(xxx/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.2.8.dylib
#  Referenced from: xxx/pyscf/lib/libcgto.dylib
#  Reason: unsafe use of relative rpath libcint.2.8.dylib in xxx/pyscf/lib/libao2mo.dylib with restricted binary
#
# It requires following fixing

dirnow=$(pwd)/$(dirname $0)

cd $dirnow
for i in *.so
do
  echo "patchelf --set-rpath '' $i${dirnow}/$i"
  patchelf --set-rpath '' $i
done
