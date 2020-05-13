#!/bin/bash
#
# This issue has been solved in PR #149.  Remove me in the future
#


dirnow=$(pwd)/$(dirname $0)

cd $dirnow
for i in *.so
do
  echo "patchelf --set-rpath '' ${dirnow}/$i"
  patchelf --set-rpath '' $i
done

echo ""
echo "RPATH has been removed.  The following paths need to be included in your LD_LIBRARY_PATH or DYLD_LIBRARY_PATH"
echo ""
echo "export LD_LIBRARY_PATH=$dirnow:\$LD_LIBRARY_PATH"
echo "export LD_LIBRARY_PATH=$dirnow/deps/lib:\$LD_LIBRARY_PATH"
echo ""
