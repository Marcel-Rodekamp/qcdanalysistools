import numpy as np

def get_block(t_data,t_block_id,t_block_size,t_is_end=False):
    if t_is_end:
        return t_data[t_block_id*t_block_size : ]
    else:
        return t_data[t_block_id*t_block_size : (t_block_id+1)*t_block_size]

def blocking_data(t_data,t_num_blocks):
    subdata_sets = [None]*t_num_blocks

    block_size = t_data.shape[0] // t_num_blocks

    # slice the data in subblocks
    for block_id in range(t_num_blocks):
        subdata_sets[block_id] = get_block(t_data,block_id,block_size)

    return np.array(subdata_sets)

def blocking_est(t_data, t_num_blocks = 2):
    blocked_data = blocking_data(t_data,t_num_blocks=t_num_blocks)

    # determine and return the estimator
    # the first (inner) average, averages in each block, the second (outer) does
    # over the subdata sets, for index details see blocking.blocking_data
    # documentation.
    return np.average( np.average(blocked_data, axis = 1), axis = 0 )

def blocking_var(t_data, t_num_blocks = 2):
    blocked_data = blocking_data(t_data,t_num_blocks=t_num_blocks)

    return np.var( np.average(blocked_data, axis = 1), axis = 0 )

def var_per_num_blocks(t_data,t_num_blocks_range = None):
    """
        t_data: numpy.ndarray
            Data which becomes blocked and processed.
        t_num_blocks_range: int, list of ints, default: None
            Determines which number of blocks are used in the process.
            * int: highest number of blocks, creating list of integers starting
                   with 2 and step 1
            * list of ints: each element is taken as a number of blocks
            * None (default): Creating a list of integers starting with 2 and
                              step 1.
        Returns: numpy.ndarray
            Array corresponding to the variance per block size. Let K be a block
            size in the list determined by t_num_blocks_range and * denotes the
            dimensionality of the estimator i.e. t_data.shape[1:], then
                var[K][*]

        This function blocks the input data in K blocks for every K in the list
        of number of blocks determined by t_num_blocks_range. Then variance is
        determined appropriately on the K blocks.
        The variances per K is returned.

        TODO:
            * This has not the desired generality
                * support for other variance estimators i.e. jackknife/bootstrap
    """
    # create list of block numbers if not given
    if t_num_blocks_range is None:
        t_num_blocks_range = [i for i in range(2,t_data.shape[0]//2)]
    elif isinstance(t_num_blocks_range,int):
        t_num_blocks_range = [i for i in range(2,t_num_blocks_range)]

    # create empty to store the variances
    var = np.zeros( shape=(len(t_num_blocks_range),*t_data.shape[1:]) )

    for num_blocks_id in range(len(t_num_blocks_range)):
        # for each number of blocks compute the blocking variance
        var[num_blocks_id] = blocking_var(t_data = t_data,
                t_num_blocks = t_num_blocks_range[num_blocks_id])

    return var

def plot_var_per_num_blocks(t_data, t_num_blocks_range = None):
    """
        t_data: numpy.array
            Data to be processed by the blocking method
        t_num_blocks_range: int, list of ints, default: None
            Determines which number of blocks are used in the process.
            * int: highest number of blocks, creating list of integers starting
                   with 2 and step 1
            * list of ints: each element is taken as a number of blocks
            * None (default): Creating a list of integers starting with 2 and
                              step 1.

        This function blocks the data with several number of blocks and for each
        the variance of the data is computed.
        The set of variances, created in this way, is then plotted and fitted with
        the A/#NumBlocks ansatz.
        The plot is not automatically saved rather only shown.

        Requirements:
            * matplotlib.pyplot
            * lmfit

        TODO:
            * This has not yet the desired generality
                * support for multidimensional estimators
                * support for non standart variances i.e. bootstrap/jackknife
    """
    import matplotlib.pyplot as plt
    import lmfit

    # create list of block numbers if not given
    if t_num_blocks_range is None:
        t_num_blocks_range = [i for i in range(2,t_data.shape[0]//2)]
    elif isinstance(t_num_blocks_range,int):
        t_num_blocks_range = [i for i in range(2,t_num_blocks_range)]

    # get the variances for the different block numbers
    vars = var_per_num_blocks(t_data,
            t_num_blocks_range = t_num_blocks_range)

    # fit the variances with A/#blockNumber
    model = lmfit.Model(lambda x,A: A/x, param_names=['A'],name="A/x")
    fit_result = model.fit(vars,x=t_num_blocks_range,A=1)
    print(fit_result.fit_report())

    # plot the data and the fit
    plt.plot(t_num_blocks_range, vars,'x',
             label="Variance(#Blocks)")
    plt.plot(t_num_blocks_range, fit_result.best_fit,
             label="{0:.1e}/#Blocks: best fit".format(fit_result.params['A'].value))
    plt.xticks(t_num_blocks_range)
    plt.grid()
    plt.xlabel("#Blocks")
    plt.ylabel("Var(#Blocks)")
    plt.legend()
    plt.show()
    # clear the plot so that other functions might not be disturbed
    plt.clf()
