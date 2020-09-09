class Refdata:

    valid_attributes = [
            "dmet_bath",
            "occ_bath",
            "vir_bath",
            "occ_bath_eigref",
            "vir_bath_eigref",
            ]

    def __init__(self, **kwargs):
        for attr in self.valid_attributes:
            setattr(self, attr, kwargs.get(attr, None))

    def __setattr__(self, name, value):
        if name not in self.valid_attributes:
            raise ValueError("Invalid attribute: %s" % name)
        setattr(self, name, value)
