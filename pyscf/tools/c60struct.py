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

from functools import reduce
import numpy

def rotmatz(ang):
    c = numpy.cos(ang)
    s = numpy.sin(ang)
    return numpy.array((( c, s, 0),
                        (-s, c, 0),
                        ( 0, 0, 1),))
def rotmaty(ang):
    c = numpy.cos(ang)
    s = numpy.sin(ang)
    return numpy.array((( c, 0, s),
                        ( 0, 1, 0),
                        (-s, 0, c),))

def r2edge(ang, r):
    return 2*r*numpy.sin(ang/2)



def make60(b5, b6):
    theta1 = numpy.arccos(1/numpy.sqrt(5))
    theta2 = (numpy.pi - theta1) * .5
    r = (b5*2+b6)/2/numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s1 = numpy.sin(theta1)
    c1 = numpy.cos(theta1)
    s2 = numpy.sin(theta2)
    c2 = numpy.cos(theta2)
    p1 = numpy.array(( s2*b5,  0, r-c2*b5))
    p9 = numpy.array((-s2*b5,  0,-r+c2*b5))
    p2 = numpy.array(( s2*(b5+b6),  0, r-c2*(b5+b6)))
    rot1 = reduce(numpy.dot, (rotmaty(theta1), rot72, rotmaty(-theta1)))
    p2s = []
    for i in range(5):
        p2s.append(p2)
        p2 = numpy.dot(p2, rot1)

    coord = []
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for pj in p2s:
        pi = pj
        for i in range(5):
            coord.append(pi)
            pi = numpy.dot(pi, rot72)
    for pj in p2s:
        pi = pj
        for i in range(5):
            coord.append(-pi)
            pi = numpy.dot(pi, rot72)
    for i in range(5):
        coord.append(p9)
        p9 = numpy.dot(p9, rot72)
    return numpy.array(coord)


def make12(b):
    theta1 = numpy.arccos(1/numpy.sqrt(5))
    theta2 = (numpy.pi - theta1) * .5
    r = b/2/numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s1 = numpy.sin(theta1)
    c1 = numpy.cos(theta1)
    p1 = numpy.array(( s1*r,  0,  c1*r))
    p2 = numpy.array((-s1*r,  0, -c1*r))
    coord = [(  0,  0,    r)]
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for i in range(5):
        coord.append(p2)
        p2 = numpy.dot(p2, rot72)
    coord.append((  0,  0,  -r))
    return numpy.array(coord)


def make20(b):
    theta1 = numpy.arccos(numpy.sqrt(5)/3)
    theta2 = numpy.arcsin(r2edge(theta1,1)/2/numpy.sin(numpy.pi/5))
    r = b/2/numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s2 = numpy.sin(theta2)
    c2 = numpy.cos(theta2)
    s3 = numpy.sin(theta1+theta2)
    c3 = numpy.cos(theta1+theta2)
    p1 = numpy.array(( s2*r,  0,  c2*r))
    p2 = numpy.array(( s3*r,  0,  c3*r))
    p3 = numpy.array((-s3*r,  0, -c3*r))
    p4 = numpy.array((-s2*r,  0, -c2*r))
    coord = []
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for i in range(5):
        coord.append(p2)
        p2 = numpy.dot(p2, rot72)
    for i in range(5):
        coord.append(p3)
        p3 = numpy.dot(p3, rot72)
    for i in range(5):
        coord.append(p4)
        p4 = numpy.dot(p4, rot72)
    return numpy.array(coord)


if __name__ == '__main__':
    b5 = 1.46
    b6 = 1.38
    for c in make60(b5, b6):
        print(c)

    b = 1.4
    for c in make12(b):
        print(c)

    for c in make20(b):
        print(c)
