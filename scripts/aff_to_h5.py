import sys
import pathlib
import aff
import h5py as h5
import numpy as np
from time import time_ns

# ==============================================================================
# Function Definition
# ==============================================================================

def recursion_keypath(t_reader,t_key,t_h5_file_handle):
    next_keypaths = t_reader.ls(t_key)

    # if an empty list apears we can read the data and return it
    if not next_keypaths:
        pathKey,dataKey = t_key.rsplit('/',1)
        if not pathKey in t_h5_file_handle:
            grp = t_h5_file_handle.create_group(pathKey)
            grp.create_dataset(dataKey,data = t_reader.read(t_key))
        else:
            t_h5_file_handle.create_dataset(pathKey+'/'+dataKey,data = t_reader.read(t_key))

    else:
        for key in next_keypaths:
            recursion_keypath(t_reader,t_key+'/'+key,t_h5_file_handle)

def convert_from_file(t_in_fn,t_out_fn):
    print(f"Converting file: {t_in_fn}\n"
        + f"to file        : {t_out_fn}")
    reader = aff.Reader(t_in_fn)

    with h5.File(t_out_fn,'w') as h5f:
        for key in reader.ls(""):
            recursion_keypath(reader,key,h5f)

    reader.close()

# ==============================================================================
# Script start
# ==============================================================================

if len(sys.argv) != 3:
    raise ValueError(f"Require exactly 2 input arguments please use: python aff_to_h5.py input_aff_fn output_h5_fn")

print(f"Reading from: {sys.argv[1]}")
print(f"Writing to  : {sys.argv[2]}")

in_arg = pathlib.Path(sys.argv[1])
out_arg = pathlib.Path(sys.argv[2])

if not in_arg.exists():
    raise ValueError("The Path/File {sys.argv[1]} does not exist.")

if not out_arg.exists():
    print("Creating output directory")
    out_arg.mkdir(parents=True,exist_ok=True)

if in_arg.is_dir():
    for fn in in_arg.iterdir():
        time_start = time_ns() * 1e-9
        convert_from_file(fn,out_arg/(fn.stem + ".h5"))
        time_end = time_ns() * 1e-9
        print(f"Converstion took {time_end-time_start:g} s")
else:
    time_start = time_ns() * 1e-9
    convert_from_file(in_arg,out_arg)
    time_end = time_ns() * 1e-9

    print(f"Converstion took {time_end-time_start:g} s")
