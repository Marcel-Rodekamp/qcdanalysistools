try:
    from .read_aff import *
except ModuleNotFoundError as e:
    print(f"Submodule qcdanalysys.fileIO.affIO not available due to exception:\n{e}")
    exit()
