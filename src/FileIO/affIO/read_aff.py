import numpy as np
from pathlib import Path
import aff

def keypath_from_list(t_key_list):
    # prepare keypath
    keypath = ""

    for key in t_key_list:
        # check that each key ends with / and has no / at start
        if key[0] == '/':
            key = key[1:]
        if key[-1] != '/':
            key+='/'

        keypath+=key

    return keypath


def read_data(t_path, t_key_list, t_auto_first_key = None):
    """
        t_path: pathlib.Path or string
            Represents the path where the *.aff data files are stored. This function
            reads all contained .aff files.
        t_key_list: list of strings
            Represents the keypath key0/key1/key2/... within the .aff files stored
            in t_path.
        t_auto_first_key: int, default: None
            If the first key, i.e. key0 is not the same for all files in t_path and
            automatically insert the one determined by
            ```
            aff.Reader.ls("")[t_auto_first_key]
            ```
            If None is given omit this step. The keypath will then be the intersection
            of all keys in `t_key_list`

        Returns: numpy.ndarray
            array representing the correlators in the format
                [ [gauge1], [gauge2], ..., [gaugeN] ]
            where [gauge1] is a numpy.array of the size t_Nt (temporal direction of gauge
            configuration used for the correlator)

        This functions reads all the data in t_path for a given set of t_keys.
        It returns a single np.ndarray containing the data.
        It is assumed that each file corresponds to one configuration.

    """
    if isinstance(t_path,str):
        t_path = Path(t_path)

    # get the key path for the aff files
    keypath = keypath_from_list(t_key_list)

    # get the aff file names
    data_fn_list = [fn.name for fn in t_path.rglob('*.aff')]


    # get the data sizes. It is assumed that all files contain the same size under keypath
    size_reader = aff.Reader(t_path/data_fn_list[0])
    if t_auto_first_key is not None:
        l_keypath = size_reader.ls("")[t_auto_first_key]+'/'+keypath
        data = np.ndarray(shape = (len(data_fn_list), size_reader.size(l_keypath)), dtype = size_reader.type(l_keypath))
    else:
        data = np.ndarray(shape = (len(data_fn_list), size_reader.size(keypath)), dtype = size_reader.type(keypath))
    size_reader.close()

    # a counter to index [gauge1],[gauge2], ...
    print("Reading files...")
    for i_fn,fn in enumerate(data_fn_list):
        # determine keypath:
        fn = t_path/fn

        # create reader
        reader = aff.Reader(fn)

        # correct the keypath bey key0 if desired
        if t_auto_first_key is not None:
            l_keypath = reader.ls("")[t_auto_first_key]+'/'+keypath
        else:
            l_keypath = keypath

        # read the data and store it in the array
        data[i_fn] = np.array( reader.read(l_keypath) )

        # finialize this step and go to next gauge file
        reader.close()

    print(f"Done reading! Read {data.shape[0]} files, with sizes = {data.shape[1:]}.")

    return data
