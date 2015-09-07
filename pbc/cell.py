import pyscf.gto

class Cell(pyscf.gto.Mole):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.h = None
        self.vol = 0.

