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
    if t_num_blocks_range is None:
        t_num_blocks_range = [i for i in range(2,t_data.shape[0]//2)]
    elif isinstance(t_num_blocks_range,int):
        t_num_blocks_range = [i for i in range(2,t_num_blocks_range)]

    var = np.zeros( shape=(len(t_num_blocks_range),*t_data.shape[1:]) )

    for num_blocks_id in range(len(t_num_blocks_range)):

        var[num_blocks_id] = blocking_var(t_data = t_data,
                t_num_blocks = t_num_blocks_range[num_blocks_id])

    return var

def plot_var_per_num_blocks(t_data, t_num_blocks_range = None):
    """
        t_var: function
            Function to evaluate the variance e.g.
                * numpy.var
                * Bootstrap
                * Jackknife
                * Blocking ...
        t_avg: function
            Function to average or compute an observable (including averages)
            Past to the blocking procedure and t_est if required.

        TODO: This has not yet the desired generality, coming soon...
    """
    import matplotlib.pyplot as plt
    import lmfit

    if t_num_blocks_range is None:
        t_num_blocks_range = [i for i in range(2,t_data.shape[0]//2)]
    elif isinstance(t_num_blocks_range,int):
        t_num_blocks_range = [i for i in range(2,t_num_blocks_range)]

    vars = var_per_num_blocks(t_data,
            t_num_blocks_range = t_num_blocks_range)

    model = lmfit.Model(lambda x,A: A/x, param_names=['A'],name="A/x")

    fit_result = model.fit(vars,x=t_num_blocks_range,A=1)

    print(fit_result.fit_report())

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
    plt.clf()
