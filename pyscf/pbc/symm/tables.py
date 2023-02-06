# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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
#
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

CrystalClass = {
    '1'     : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '-1'    : [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    '2'     : [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'm'     : [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    '2/m'   : [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    '222'   : [0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
    'mm2'   : [0, 0, 0, 2, 0, 1, 1, 0, 0, 0],
    'mmm'   : [0, 0, 0, 3, 1, 1, 3, 0, 0, 0],
    '4'     : [0, 0, 0, 0, 0, 1, 1, 0, 2, 0],
    '-4'    : [0, 2, 0, 0, 0, 1, 1, 0, 0, 0],
    '4/m'   : [0, 2, 0, 1, 1, 1, 1, 0, 2, 0],
    '422'   : [0, 0, 0, 0, 0, 1, 5, 0, 2, 0],
    '4mm'   : [0, 0, 0, 4, 0, 1, 1, 0, 2, 0],
    '-42m'  : [0, 2, 0, 2, 0, 1, 3, 0, 0, 0],
    '4/mmm' : [0, 2, 0, 5, 1, 1, 5, 0, 2, 0],
    '3'     : [0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
    '-3'    : [0, 0, 2, 0, 1, 1, 0, 2, 0, 0],
    '32'    : [0, 0, 0, 0, 0, 1, 3, 2, 0, 0],
    '3m'    : [0, 0, 0, 3, 0, 1, 0, 2, 0, 0],
    '-3m'   : [0, 0, 2, 3, 1, 1, 3, 2, 0, 0],
    '6'     : [0, 0, 0, 0, 0, 1, 1, 2, 0, 2],
    '-6'    : [2, 0, 0, 1, 0, 1, 0, 2, 0, 0],
    '6/m'   : [2, 0, 2, 1, 1, 1, 1, 2, 0, 2],
    '622'   : [0, 0, 0, 0, 0, 1, 7, 2, 0, 2],
    '6mm'   : [0, 0, 0, 6, 0, 1, 1, 2, 0, 2],
    '-6m2'  : [2, 0, 0, 4, 0, 1, 3, 2, 0, 0],
    '6/mmm' : [2, 0, 2, 7, 1, 1, 7, 2, 0, 2],
    '23'    : [0, 0, 0, 0, 0, 1, 3, 8, 0, 0],
    'm-3'   : [0, 0, 8, 3, 1, 1, 3, 8, 0, 0],
    '432'   : [0, 0, 0, 0, 0, 1, 9, 8, 6, 0],
    '-43m'  : [0, 6, 0, 6, 0, 1, 3, 8, 0, 0],
    'm-3m'  : [0, 6, 8, 9, 1, 1, 9, 8, 6, 0],
}

LaueClass = {
    '-1'    : ['1', '-1'],
    '2/m'   : ['2', 'm', '2/m'],
    'mmm'   : ['222', 'mm2', 'mmm'],
    '4/m'   : ['4', '-4', '4/m'],
    '4/mmm' : ['422', '4mm', '-42m', '4/mmm'],
    '-3'    : ['3', '-3'],
    '-3m'   : ['32', '3m', '-3m'],
    '6/m'   : ['6', '-6', '6/m'],
    '6/mmm' : ['622', '6mm', '-6m2', '6/mmm'],
    'm-3'   : ['23', 'm-3'],
    'm-3m'  : ['432', '-43m', 'm-3m'],
}

SchoenfliesNotation = {
    '1'     : 'C1',
    '-1'    : 'Ci',
    '2'     : 'C2',
    'm'     : 'Cs',
    '2/m'   : 'C2h',
    '222'   : 'D2',
    'mm2'   : 'C2v',
    'mmm'   : 'D2h',
    '4'     : 'C4',
    '-4'    : 'S4',
    '4/m'   : 'C4h',
    '422'   : 'D4',
    '4mm'   : 'C4v',
    '-42m'  : 'D2d',
    '4/mmm' : 'D4h',
    '3'     : 'C3',
    '-3'    : 'S6',
    '32'    : 'D3',
    '3m'    : 'C3v',
    '-3m'   : 'D3d',
    '6'     : 'C6',
    '-6'    : 'C3h',
    '6/m'   : 'C6h',
    '622'   : 'D6',
    '6mm'   : 'C6v',
    '-6m2'  : 'D3h',
    '6/mmm' : 'D6h',
    '23'    : 'T',
    'm-3'   : 'Th',
    '432'   : 'O',
    '-43m'  : 'Td',
    'm-3m'  : 'Oh',
}
