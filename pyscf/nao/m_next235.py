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

def next235(base):
  #assert(type(base)==float)
  next235 = 2 * int(base/2.0+.9999)
  if (next235<=0) : next235 = 2
  while 100000:
    numdiv = next235
    while ((numdiv//2)*2 == numdiv): numdiv = numdiv//2
    while ((numdiv//3)*3 == numdiv): numdiv = numdiv//3
    while ((numdiv//5)*5 == numdiv): numdiv = numdiv//5
    if numdiv == 1: return next235
    next235 = next235 + 2
  raise RuntimeError('too difficult to find...')
   
  
