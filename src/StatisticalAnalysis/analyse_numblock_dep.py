import qcdanalysistools.fitting as fitting
from . import variance
from . import Blocking,AnalysisParam
import matplotlib.pyplot as plt
import numpy as np

def analyse_numblock_dependency(t_data, t_Nbl_list, t_fn = "./blocking_analysis.pdf", t_var_factor = 1):
    r"""

    """
    color_list=['tab:blue','tab:orange','tab:green','tab:red','tab:purple',"tab:brown","tab:pink","tab:gray","tab:olive",'tab:cyan'] # 10 colors

    t_Nbl_list = np.array(t_Nbl_list)
    block_sizes = t_data.shape[0]//t_Nbl_list

    var_per_Nbl = np.zeros(shape=(t_Nbl_list.shape[0],*t_data.shape[1:]))

    for i_bl,N_bl in enumerate(t_Nbl_list):
        bl_params = AnalysisParam(Blocking,
            data_size  = t_data.shape[0],
            N_blk      = N_bl
        )
        var_per_Nbl[i_bl,:] = variance(bl_params,t_data,t_axis=0)*t_var_factor

    # Fit the variance to A/num_blocks
    fit_model = fitting.PropInversModel(t_A0=1.7)
    fit_results = []
    for i in range(len(t_Nbl_list) - 2):

        fitter = fitting.DiagonalLeastSquare(
            t_model=fit_model,
            t_abscissa=block_sizes[i:],
            t_ordinate=var_per_Nbl[i:,0],
            t_ordinate_var=np.ones_like(var_per_Nbl[i:,0])
        )

        fit_results.append(fitter.fit())
        print(f"Fitting from {t_Nbl_list[i]} to {t_Nbl_list[-1]}")
        fitter.print_result()

    fig = plt.figure(figsize=(11.7,8.3))

    ax_var = fig.add_subplot(2,2,1)
    ax_err = fig.add_subplot(2,2,2)
    ax_chi = fig.add_subplot(2,2,3)
    ax_pva = fig.add_subplot(2,2,4)

    dat_err = np.zeros(len(fit_results))
    dat_chi = np.zeros(len(fit_results))
    dat_pva = np.zeros(len(fit_results))

    # ==========================================================================
    # Variance by N_blocks
    # ==========================================================================
    for i_fr,fit_res in enumerate(fit_results):
        if i_fr == 0:
            ax_var.plot(block_sizes,fit_res['Best fit'],':',c=color_list[0],label="Best fits")
        else:
            ax_var.plot(block_sizes[i_fr:],fit_res['Best fit'],':',c=color_list[0])
            #ax_var.errorbar(t_Nbl_list[i_fr:],fit_res['Best fit'],yerr=np.ones_like(fit_res['Best fit'])*fit_res['Fit error'],fmt=':',c=color_list[0])

        dat_err[i_fr] = fit_res['Fit error']
        dat_chi[i_fr] = fit_res['red chisq']
        dat_pva[i_fr] = fit_res['p-value']
    ax_var.plot(block_sizes,var_per_Nbl[:,0],'o:',c=color_list[3],label="Var")

    ax_var.set_xlabel(r"Block Size")
    if t_var_factor == 1:
        ax_var.set_ylabel(r"$Var$")
    else:
        ax_var.set_ylabel(fr"$Var\cdot {t_var_factor:g}$")
    ax_var.legend()
    #ax_var.set_xticks(block_sizes[::4])

    # ==========================================================================
    # Fit error by N_blocks
    # ==========================================================================
    ax_err.plot(block_sizes[:-2],dat_err,'x:')
    ax_err.set_xlabel(r"Block Size")
    ax_err.set_ylabel(r"Fit error")
    #ax_err.set_xticks(block_sizes[::4])
    # ==========================================================================
    # Chisq/dof by N_blocks
    # ==========================================================================
    ax_chi.plot(block_sizes[:-2],dat_chi,'x:')
    ax_chi.set_yscale('log')
    ax_chi.set_xlabel(r"Block Size")
    ax_chi.set_ylabel(r"$\frac{\chi^2}{dof}$")
    #ax_chi.set_xticks(block_sizes[::4])

    # ==========================================================================
    # p-value by N_blocks
    # ==========================================================================
    ax_pva.plot(block_sizes[:-2],dat_pva,'x:')
    ax_pva.set_xlabel(r"Block Size")
    ax_pva.set_ylabel(r"p-value")
    #ax_pva.set_xticks(block_sizes[::4])

    fig.tight_layout()
    plt.savefig(t_fn)
    plt.clf()
    plt.close()
