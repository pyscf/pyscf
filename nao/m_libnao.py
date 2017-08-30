from pyscf.lib import misc
libnao = misc.load_library("libnao")
try:
    libnao_gpu = misc.load_library("libnao_gpu")
except:
    print("Failed to import libnao_gpu")
