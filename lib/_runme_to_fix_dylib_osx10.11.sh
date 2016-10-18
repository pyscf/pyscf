#!/bin/bash
# On Mac OSX 10.11 or newer, you may get error message if you use clang compiler
#
#  OSError: dlopen(xxx/pyscf/lib/libcgto.dylib, 6): Library not loaded: libcint.2.8.dylib
#  Referenced from: xxx/pyscf/lib/libcgto.dylib
#  Reason: unsafe use of relative rpath libcint.2.8.dylib in xxx/pyscf/lib/libao2mo.dylib with restricted binary
#
# It requires following fixing

dirnow=$(pwd)
for i in libao2mo.dylib libcc.dylib libcgto.dylib libcvhf.dylib libdft.dylib \
  libfci.dylib libicmpspt.dylib liblocalizer.dylib libmcscf.dylib \
  libnp_helper.dylib libpbc.dylib libri.dylib libxcfun_itrf.dylib \
  libxc_itrf.dylib
do
  install_name_tool -change libcint.2.8.dylib $(dirnow)/pyscf/lib/deps/lib/libcint.2.8.dylib $(dirnow)/pyscf/lib/$i
done
