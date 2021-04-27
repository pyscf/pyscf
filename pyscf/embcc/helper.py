
default_minao = {
        "gth-dzv" : "gth-szv",
        "gth-dzvp" : "gth-szv",
        "gth-tzvp" : "gth-szv",
        "gth-tzv2p" : "gth-szv",
        }

def get_minimal_basis(basis):
    minao = default_minao.get(basis, "minao")
    return minao
