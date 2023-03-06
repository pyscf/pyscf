import numpy
from pyscf import gto, dft


if __name__ == '__main__':
    water = gto.M(atom='O 0 0 0; H 0.757541 0.587080 0.0; H -0.757541 0.5870080 0.0', basis='aug-cc-pvtz')
    h2s = gto.M(atom='S 0 0 0; H 0.961847 0.927225 0; H -0.961847 0.927225 0', basis='aug-cc-pvtz')

    mol = h2s

    rks = dft.RKS(mol)
    rks.xc = 'wb97x-d3'
    rks.run()

    dm = rks.make_rdm1()

    #grids = rks_water.grids # grids.coords  shape (N, 3) x y z
    #indx1 = grids.coords[numpy.where(grids.coords[:,2]==0)]

    xg = numpy.linspace(-0.1, 0.1, 100)
    yg = numpy.linspace(-0.1, 0.1, 100)
    zg = numpy.zeros(1)
    indx = numpy.array(numpy.meshgrid(xg, yg, zg))[:,:,:,0]
    indx = numpy.einsum('ijk->jki', indx)

    xgrid = indx[:,:,0].reshape(len(xg)**2)
    ygrid = indx[:,:,1].reshape(len(yg)**2)
    zgrid = indx[:,:,2].reshape(len(xg)**2)

    indx = numpy.zeros((len(xgrid), 3))
    indx[:,0] = xgrid
    indx[:,1] = ygrid
    indx[:,2] = zgrid

    ao = dft.numint.eval_ao(mol, indx) # (N, nao)
    rho = dft.numint.eval_rho(mol, ao, dm ) # (N, rho)

    numpy.set_printoptions(linewidth=1000)

    width = 50
    cursor = 0
    string = ""

    for i in range(len(rho)):
        string += str(round(rho[i], 10)) + ";"
        if float(i+1) % float(width) < 1:
            print(string)
            string = ""

    if len(string) > 1:
        print(string)



