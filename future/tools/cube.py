import numpy


def make111(a, rads, pactive=None, plainvec=(1,1,1), pt0=(0,0,0)):
    pt0 = numpy.array(pt0)
    plain = numpy.array(plainvec)/numpy.linalg.norm(plainvec)
    if pactive is None:
        pactive = pt0 + plain
    atms = []
    for i in range(-8,8):
        for j in range(-8,8):
            k0 = (pt0[2] - ((i*a-pt0[0])*plain[0]+(j*a-pt0[1])*plain[1])/plain[2])/a
            for k in range(-8,int(k0)+1):
                pt1 = numpy.array((i,j,k)) * a
                rr = numpy.linalg.norm(pt1 - pactive)
                for nr, ri in enumerate(rads):
                    if rr < ri:
                        atms.append((nr,pt1))
                        break
    return atms

def make100():
    pass

def make110():
    pass

if __name__ == '__main__':
    for i,a in enumerate(make111(1, (1.1,2,3.0), (.6,.6,.6))):
        print(i, a)
