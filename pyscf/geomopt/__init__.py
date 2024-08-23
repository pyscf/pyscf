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

#from . import berny_solver as berny

from .addons import as_pyscf_method

def optimize(method, *args, **kwargs):
    try:
        from . import geometric_solver as geom
    except ImportError as e1:
        try:
            from . import berny_solver as geom
        except ImportError:
            raise e1
    return geom.optimize(method, *args, **kwargs)
