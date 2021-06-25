import unittest
import itertools as it
import qcdanalysistools as tools
import numpy as np

# distribution details:
# - uniform [a,b)
#   * https://en.wikipedia.org/wiki/Continuous_uniform_distribution
#   * mean 0.5 (a+b)
#   * var = 1/12 (b-a)^2
# - beta(a,b)
#   * https://en.wikipedia.org/wiki/Beta_distribution
#   * mean a/(a+b)
#   * var = a*b/((a+b+1)(a+b^2))
# - binomial(n,p)
#   * https://en.wikipedia.org/wiki/Binomial_distribution
#   * mean np
#   * var = np*(1-p)


class TestBootstrap(unittest.TestCase):
    def _obs_1(self,data):
        # some test observable without parameter
        return np.exp(data)

    def _obs_2(self,data, axis):
        # some test observable reducing given axis
        return np.mean(self._obs_1(data),axis=axis)

    def testDimensionalities(self):
        """
            This test creates data uniformly on [0,1) for several input dimensions
            On each of these datas estimator and variances are computed with both
            interfaces, single est & var and combined one. Then the outcome array
            dimension is checked for several use cases such as
                * Estimate over one axis
                * Estimate over one axis after application of observable (self._obs_1)
                * Estimate over 0 axis after application of observable reducing last dimension (self._obs_2(data,axis=-1))
                * Estimate over 0 axis after application of observable reducing all but the first dimension (self._obs_2(data,axis=(1,2,...dim-1)))
        """
        for dim in [1,2,3,4,5,6]:
            # datasize
            N = tuple([10]*dim)

            # data: uniform [0,1)
            data = np.random.rand(*N)

            #==================================================
            # No additional reduction in dimension by observables
            #==================================================
            for ax,N_bst in it.product(range(dim),range(1,100,10)):
                # create Analysis parameters
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = 10,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = ax
                )
                # single interface
                est_1 = tools.analysis.estimator(t_param=param,t_data=data)
                est_2 = tools.analysis.estimator(t_param=param,t_data=data,t_observable=self._obs_1)
                var_1 = tools.analysis.variance(t_param=param,t_data=data)
                var_2 = tools.analysis.variance(t_param=param,t_data=data,t_observable=self._obs_1)
                self.assertEqual(est_1.shape,tuple([10]*(dim-1)))
                self.assertEqual(var_2.shape,tuple([10]*(dim-1)))
                # combined interface
                est_1,var_1 = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                est_2,var_2 = tools.analysis.dataAnalysis(t_param=param,t_data=data,t_observable=self._obs_1)
                self.assertEqual(est_2.shape,tuple([10]*(dim-1)))
                self.assertEqual(var_1.shape,tuple([10]*(dim-1)))

            #==================================================
            # Additional reduction in dimension by observables
            #==================================================
            # skip for 1D arrays
            if dim == 1:
                continue

            #==========================
            # Reduce last dimension
            #==========================
            for N_bst in range(1,100,10):
                # create Analysis parameters
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = 10,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = 0 # that's the default
                )

                # single interface
                est = tools.analysis.estimator(t_param=param,t_data=data,t_observable=self._obs_2,axis=-1)
                var = tools.analysis.variance(t_param=param,t_data=data,t_observable=self._obs_2,axis=-1)
                self.assertEqual(est.shape,tuple([10]*(dim-2)))
                self.assertEqual(var.shape,tuple([10]*(dim-2)))
                # combined interface
                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data,t_observable=self._obs_2,axis=-1)
                self.assertEqual(est.shape,tuple([10]*(dim-2)))
                self.assertEqual(var.shape,tuple([10]*(dim-2)))

            #==========================
            # Reduce all dimensions dimension
            #==========================
            # 0 is reduced by analysis, 1,2,...dim is reduced by _obs_2
            ax = tuple([i for i in range(1,dim)])
            for N_bst in range(1,100,10):
                # create Analysis parameters
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = 10,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = 0 # that's the default
                )

                # single interface
                est = tools.analysis.estimator(t_param=param,t_data=data,t_observable=self._obs_2,axis=ax)
                var = tools.analysis.variance(t_param=param,t_data=data,t_observable=self._obs_2,axis=ax)
                self.assertEqual(est.shape,())
                self.assertEqual(var.shape,())
                # combined interface
                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data,t_observable=self._obs_2,axis=ax)
                self.assertEqual(est.shape,())
                self.assertEqual(var.shape,())

    def testUniform(self):
        """
            Test that mean of Uniform distribution is found correctly
            Note this is uncorrelated data!
        """
        for N in range(100,500,100):
            print(f"Testing Estimation of uniform distribution for a data size of {N}.")
            data = np.random.uniform(low=0,high=1,size=(N,))

            print("Non blocking...", end=" ")
            for N_bst in range(10,500,10):
                # non blocking
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # Number of bootstrap samples
                    N_bst = N_bst,
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    axis = 0,
                    # we can store the indices associated with the bst using h5
                    store_bst_samples = False,
                    # in that case we can provide a file name
                    store_bst_samples_fn = "./boostrap_samples.h5", # that'd be the default
                )

                # check if the true value (0.5) is in the intervall est +/- err
                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

            print("done\nBlocking...", end=" ")
            for N_bst,N_blk in it.product(range(10,500,10), range(2,N//2,10)):
                # blocking
                # corner case can not be handled. But simplifies loop so just continue
                if N_bst >= N//N_blk:
                    continue

                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = True,
                    # block size irrelevant if use_blocking = False
                    N_blk = N_blk,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = 0
                )

                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))
            print("done")

    def testBeta(self):
        """
            Test that mean of Beta distribution is found correctly
            Note this is uncorrelated data!
        """
        for N in range(100,500,100):
            print(f"Testing Estimation of Beta distribution for a data size of {N}.")
            data = np.random.beta(a=1,b=1,size=(N,))

            print("Non blocking...", end=" ")
            for N_bst in range(10,500,10):
                # non blocking
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # Number of bootstrap samples
                    N_bst = N_bst,
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    axis = 0,
                    # we can store the indices associated with the bst using h5
                    store_bst_samples = False,
                    # in that case we can provide a file name
                    store_bst_samples_fn = "./boostrap_samples.h5", # that'd be the default
                )

                # check if the true value (0.5) is in the intervall est +/- err
                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

            print("done\nBlocking...", end=" ")
            for N_bst,N_blk in it.product(range(10,500,10), range(2,N//2,10)):
                # blocking
                # corner case can not be handled. But simplifies loop so just continue
                if N_bst >= N//N_blk:
                    continue

                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = True,
                    # block size irrelevant if use_blocking = False
                    N_blk = N_blk,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = 0
                )

                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))
            print("done")

    def testBinomial(self):
        """
            Test that mean of Binomial distribution is found correctly
            Note this is uncorrelated data!
        """
        for N in range(100,500,100):
            print(f"Testing Estimation of Beta distribution for a data size of {N}.")
            data = np.random.binomial(n=1,p=0.5,size=(N,))

            print("Non blocking...", end=" ")
            for N_bst in range(10,500,10):
                # non blocking
                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # Number of bootstrap samples
                    N_bst = N_bst,
                    # use blocking
                    use_blocking = False,
                    # block size irrelevant if use_blocking = False
                    N_blk = None,
                    # specify the axis over which the estimate should be taken.
                    axis = 0,
                    # we can store the indices associated with the bst using h5
                    store_bst_samples = False,
                    # in that case we can provide a file name
                    store_bst_samples_fn = "./boostrap_samples.h5", # that'd be the default
                )

                # check if the true value (0.5) is in the intervall est +/- err
                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

            print("done\nBlocking...", end=" ")
            for N_bst,N_blk in it.product(range(10,500,10), range(2,N//2,10)):
                # blocking
                # corner case can not be handled. But simplifies loop so just continue
                if N_bst >= N//N_blk:
                    continue

                param = tools.analysis.AnalysisParam(tools.analysis.Bootstrap,
                    # size of the data we are going to handle
                    data_size = N,
                    # number of elements removed i.e. leave N_bst out Bootstrap
                    N_bst = N_bst, # that's default
                    # use blocking
                    use_blocking = True,
                    # block size irrelevant if use_blocking = False
                    N_blk = N_blk,
                    # specify the axis over which the estimate should be taken.
                    # bahaviour is the same as for numpy
                    axis = 0
                )

                est = tools.analysis.estimator(t_param=param,t_data=data)
                var = tools.analysis.variance(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))

                est,var = tools.analysis.dataAnalysis(t_param=param,t_data=data)
                self.assertTrue(est - np.sqrt(var) < 0.5 or 0.5 < est + np.sqrt(var))
            print("done")

if __name__ == '__main__':
    unittest.main()
