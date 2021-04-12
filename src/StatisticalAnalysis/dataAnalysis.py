import numpy as np
import warnings
import h5py as h5
import pathlib
import itertools

class Bootstrap:
    def __name__(self):
        return "Bootstrap"
class Jackknife:
    def __name__(self):
        return "Jackknife"
class Blocking:
    def __name__(self):
        return "Blocking"

def checkAnalysisType(t_AnalysisType,t_desiredAnalysisType):
    return isinstance(t_AnalysisType(),t_desiredAnalysisType)

class AnalysisParam(dict):
    __slots__ = ('AnalysisType')

    def _checkType(self,key,t_type):
        my_val = self.get(key)
        if type(my_val) is not t_type:
            try:
                self.__setitem__(key,t_type(my_val))
            except:
                raise ValueError(f"Can't convert '{key}' to {t_type}, you passed '{my_val}' of type {type(my_val)}")

    def write_to_h5(self,h5f):
        grp = h5f.create_group("SampleParams")

        for key in self.keys():
            if key == 'store_bst_samples_fn':
                continue
            if key == 'store_bst_samples':
                continue
            grp.create_dataset(key,data=self.get(key))

    def _bst_init(self,**kwargs):
        if 'N_bst' not in kwargs:
            raise AttributeError(f"Didn't find 'N_bst' required for {self.__name__()}")
        else:
            self._checkType('N_bst',int)

        if 'use_blocking' not in kwargs:
            self.setdefault('use_blocking', default=False)

        if self.get('use_blocking'):
            if 'N_blk' not in kwargs:
                raise AttributeError(f"Didn't find 'N_blk' required for {self.__name__()} with 'use_blocking' set to True")

            self._checkType('N_blk',int)

            # set block_size
            self.__setitem__('blk_size', self.get('data_size')//self.get('N_blk'))

        if 'store_bst_samples' not in kwargs:
            self.setdefault('store_bst_samples', default=False)
        else:
            if 'store_bst_samples_fn' not in kwargs:
                self.setdefault('store_bst_samples_fn', pathlib.Path("./boostrap_samples.h5"))
            else:
                self._checkType('store_bst_samples_fn',pathlib.Path)
                # create the path if it not exist
                self.get('store_bst_samples_fn').parent.mkdir(parents=True, exist_ok=True)

            # overwrite existing file with parameter meta data
            with h5.File(self.get('store_bst_samples_fn'),'w') as h5f:
                self.write_to_h5(h5f)

    def _jak_init(self,**kwargs):
        if 'N_jak' not in kwargs:
            self.setdefault('N_jak',1)
        else:
            self._checkType('N_jak',int)

        if 'use_blocking' not in kwargs:
            self.setdefault('use_blocking', default=False)

        if self.get('use_blocking'):
            if 'N_blk' not in kwargs:
                raise AttributeError(f"Didn't find 'N_blk' required for {self.__name__()} with 'use_blocking' set to True")

            self._checkType('N_blk',int)

            # set block_size
            self.__setitem__('blk_size', self.get('data_size')//self.get('N_blk'))

    def _blk_init(self,**kwargs):
        if 'N_blk' not in kwargs:
            raise AttributeError(f"Didn't find 'N_blk' required for {self.__name__()}")
        else:
            self._checkType('N_blk',int)

        # set block_size
        self.__setitem__('blk_size', self.get('data_size')//self.get('N_blk'))

    def __init__(self,t_type,**kwargs):
        self.AnalysisType = t_type
        super(AnalysisParam,self).__init__(**kwargs)
        #print(f"args passed = {kwargs}")

        if 'data_size' not in kwargs:
            raise AttributeError(f"Didn't find 'data_size' required for {self.__name__()}")
        else:
            self._checkType('data_size',int)

        if checkAnalysisType(self.AnalysisType,Bootstrap):
            self._bst_init(**kwargs)
        elif checkAnalysisType(self.AnalysisType,Jackknife):
            self._jak_init(**kwargs)
        elif checkAnalysisType(self.AnalysisType,Blocking):
            self._blk_init(**kwargs)
        else:
            raise ValueError(f"Unknown AnalysisType: {self.AnalysisType}")

    def __getitem__(self, key):
        return super(AnalysisParam,self).__getitem__(key)

    def __setitem__(self, key, val):
        return super(AnalysisParam,self).__setitem__(key, val)

    def __delitem__(self, key):
        return super(AnalysisParam,self).__delitem__(key)

    def get(self, key, default=None):
        return super(AnalysisParam,self).get(key, default)

    def setdefault(self, key, default=None):
        return super(AnalysisParam,self).setdefault(key, default)

    def pop(self, key, val=object()):
        if val is _RaiseKeyError:
            return super(AnalysisParam,self).pop(key)
        return super(AnalysisParam,self).pop(key, val)

    def update(self,**kwargs):
        super(AnalysisParam,self).update(**kwargs)

    def __contains__(self, key):
        return super(AnalysisParam, self).__contains__(key)

    def copy(self):
        return type(self)(self)

    @classmethod
    def fromkeys(cls, keys, val=None):
        return super(AnalysisParam, cls).fromkeys((key for key in keys), val)

    def __repr__(self):
        return f"{self.__name__()}({super(AnalysisParam, self).__repr__()})"

    def __name__(self):
        return f"AnalysisParam[{self.AnalysisType.__name__}]"

    def num_samples(self):
        if checkAnalysisType(self.AnalysisType,Bootstrap):
            return self.get('N_bst')
        elif checkAnalysisType(self.AnalysisType,Jackknife):
            return self.get('data_size')
        elif checkAnalysisType(self.AnalysisType,Blocking):
            return self.get('N_blk')
        else:
            raise NotImplementedError

def get_sample(t_param,t_data,t_k,t_blk_k=None,t_axis=0):
    def _get_bst_sample(t_data):
        sample = np.take(
            t_data,
            np.random.randint(0,high=t_data.shape[t_axis],size=(t_data.shape[t_axis],) ),
            axis=t_axis
        )

        if t_param['store_bst_samples']:
            with h5.File(t_param['store_bst_samples_fn'], 'a') as h5f:
                if t_param['use_blocking']:
                    try:
                        sample_group = h5f.create_group(f"blocking_id_{t_blk_k}")
                    except:
                        sample_group = h5f[f"blocking_id_{t_blk_k}"]

                        sample_group.create_dataset(f"{t_k}",data=sample)
                else:
                    h5f.create_dataset(f"{t_k}",data=sample)

        return sample

    def _get_jak_sample(t_data):
        sample = np.delete(
                t_data,
                [t_k+i if t_k+i<t_param.get('data_size') else i for i in range(t_param.get('N_jak'))],
                axis=t_axis
        )

        return sample

    def _get_blk_sample(t_data,t_k):
        if t_k == t_param.get('N_blk') - 1:
            sample = np.take(t_data,range(t_k*t_param['blk_size'],t_data.shape[t_axis]),axis=t_axis)
        else:
            sample = np.take(t_data,range(t_k*t_param['blk_size'],(t_k+1)*t_param['blk_size']),axis=t_axis)

        return sample

    # ==========================================================================
    # Bootstrap
    # ==========================================================================
    if checkAnalysisType(t_param.AnalysisType,Bootstrap):
        if t_param['use_blocking']:
            return _get_bst_sample(t_data=_get_blk_sample(t_data=t_data,t_k=t_blk_k))
        else:
            if t_blk_k is not None:
                warnings.warn(f"You passed 't_blk_k' = {t_blk_k} to {t_param.__name__()}.get_sample without setting 'use_blocking' to True!")

            return _get_bst_sample(t_data=t_data)

    # ==========================================================================
    # Jackknife
    # ==========================================================================
    if checkAnalysisType(t_param.AnalysisType,Jackknife):
        if t_param['use_blocking']:
            return _get_jak_sample(t_data=_get_blk_sample(t_data=t_data,t_k=t_blk_k))
        else:
            if t_blk_k is not None:
                warnings.warn(f"You passed 't_blk_k' = {t_blk_k} to {t_param.__name__()}.get_sample without setting 'use_blocking' to True!")

            return _get_jak_sample(t_data=t_data)

    # ==========================================================================
    # Blocking
    # ==========================================================================
    if checkAnalysisType(t_param.AnalysisType,Blocking):
            return _get_blk_sample(t_data=t_data,t_k=t_k)

def estimator(t_param,t_data,t_observable=np.average,t_axis=0,**kwargs):
    def _use_blocking_est(N_blk,K):
        estimators = [[None]*K]*N_blk

        for k_blk,k in itertools.product(range(N_blk),range(K)):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k      ,
                t_blk_k = k_blk  ,
                t_axis  = t_axis
            )

            try:
                estimators[k_blk][k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k_blk][k] = t_observable(sample,**kwargs)

        # average over all blocking and bootstrap
        return np.average(np.average(estimators,axis=1),axis=0)

    def _est(K):
        estimators = [None]*K

        for k in range(K):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k      ,
                t_axis  = t_axis
            )

            try:
                estimators[k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k] = t_observable(sample,**kwargs)

        # average over all blocking and bootstrap
        return np.average(estimators,axis=0)

    if t_param.get('use_blocking',False):
        return _use_blocking_est(t_param['N_blk'],t_param.num_samples())
    else:
        return _est(t_param.num_samples())

def variance(t_param,t_data,t_observable=np.average,t_axis=0,**kwargs):
    def _use_blocking_var(N_blk,K):
        # Creat matrix to store the by block and bootstrap sample estimators
        # we may not know which dimensionality comes out of t_obs thus we
        # need to use python built=in list and append
        estimators = [[None]*K]*N_blk

        for k_blk,k in itertools.product(range(N_blk),range(K)):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k  ,
                t_blk_k = k_blk  ,
                t_axis  = t_axis
            )

            try:
                estimators[k_blk][k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k_blk][k] = t_observable(sample,**kwargs)

        # compute the variance on analysis samples
        variances = np.var(estimators,axis=1)
        # average the variance over the blocks
        variances = np.average(variances,axis=0)

        return variances

    def _var(K):
        # Creat vector to store the by bootstrap sample estimators
        # we may not know which dimensionality comes out of t_obs thus we
        # need to use python built=in list and append
        estimators = [None]*K

        for k in range(K):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k      ,
                t_axis  = t_axis
            )

            try:
                estimators[k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k] = t_observable(sample,**kwargs)

        return np.var(estimators,axis=0)

    if t_param.get('use_blocking',False):
        return _use_blocking_var(t_param['N_blk'],t_param.num_samples())
    else:
        return _var(t_param.num_samples())

def dataAnalysis(t_param,t_data,t_observable=np.average,t_axis=0,**kwargs):
    def _use_blocking_analysis(N_blk,K):
        # Creat matrix to store the by block and bootstrap sample estimators
        # we may not know which dimensionality comes out of t_obs thus we
        # need to use python built=in list and append
        estimators = [[None]*K]*N_blk

        for k_blk,k in itertools.product(range(N_blk),range(K)):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k  ,
                t_blk_k = k_blk  ,
                t_axis  = t_axis
            )

            try:
                estimators[k_blk][k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k_blk][k] = t_observable(sample,**kwargs)

        return np.average(np.average(estimators,axis=1),axis=0),np.average(np.var(estimators,axis=1),axis=0)

    def _analysis(K):
        # Creaet vector to store the by bootstrap sample estimators
        # we may not know which dimensionality comes out of t_obs thus we
        # need to use python built-in list and append
        estimators = [None]*K

        for k in range(K):
            sample = get_sample(
                t_param = t_param,
                t_data  = t_data ,
                t_k     = k      ,
                t_axis  = t_axis
            )

            try:
                estimators[k] = t_observable(sample,axis=t_axis,**kwargs)
            except TypeError:
                estimators[k] = t_observable(sample,**kwargs)

        if checkAnalysisType(t_param.AnalysisType,Jackknife):
            # use the correct normalization for the Jackknife variance
            return np.average(estimators,axis=0), np.var(estimators,axis=0) * (t_param['data_size']-1)
        else:
            return np.average(estimators,axis=0), np.var(estimators,axis=0)

    if t_param.get('use_blocking',False):
        return _use_blocking_analysis(t_param['N_blk'],t_param.num_samples())
    else:
        return _analysis(t_param.num_samples())
