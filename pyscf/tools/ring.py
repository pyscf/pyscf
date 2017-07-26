import numpy

def make(nat, b=1.):
    r = b/2 / numpy.sin(numpy.pi/nat)
    atms = []
    for i in range(nat):
        theta = i * (2*numpy.pi/nat)
        atms.append((r*numpy.cos(theta), r*numpy.sin(theta), 0))
    return atms

if __name__ == '__main__':
    for c in make(6):
        print(c)

