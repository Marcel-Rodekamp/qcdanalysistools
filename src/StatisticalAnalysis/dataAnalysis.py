import numpy as np
import warnings
import h5py as h5
import pathlib
import itertools

# ==============================================================================
# Parameter
# ==============================================================================

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
    __slots__ = ('AnalysisType','bst_table')

    def _checkType(self,key,t_type):
        my_val = self.get(key)
        if type(my_val) is not t_type:
            try:
                self.__setitem__(key,t_type(my_val))
            except:
                raise ValueError(f"Can't convert '{key}' to {t_type}, you passed '{my_val}' of type {type(my_val)}")

    def write_to_h5(self,h5f):

        h5f.create_dataset('bst_table',data=self.bst_table)
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
            bst_max = self.get('blk_size')
        else:
            bst_max = self.get('data_size')

        # self.bst_table = np.zeros(shape=( self.get('N_bst'), self.get('data_size')), dtype=int)
        self.bst_table = np.zeros(shape=( self.get('N_bst'), bst_max), dtype=int)
        for k in range(self.get('N_bst')):
            # self.bst_table[k,:] = np.random.randint(0,high=self.get('data_size'),size=self.get('data_size'))
            self.bst_table[k,:] = np.random.randint(0,high=bst_max,size=bst_max)

        if 'store_bst_samples' not in kwargs:
            self.setdefault('store_bst_samples', default=False)

        if self.get('store_bst_samples'):
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

            if self.get('N_jak') >= self.get('blk_size'):
                raise AttributeError(f"Can not have N_jak ({self.get('N_jak')}) >= blk_size ({self.get('data_size')}), as it would delete all elements in a block.")

    def _blk_init(self,**kwargs):
        if 'N_blk' not in kwargs:
            raise AttributeError(f"Didn't find 'N_blk' required for {self.__name__()}")
        else:
            self._checkType('N_blk',int)

        # set block_size
        self.__setitem__('blk_size', self.get('data_size')//self.get('N_blk'))

    def __init__(self,t_type,**kwargs):
        self.AnalysisType = t_type
        self.bst_table = None
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

        # contrary to the numpy ansatz set the default data axis to 0
        if 'axis' not in kwargs:
            self.setdefault('axis',0)

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
            if self.get('use_blocking'):
                return self.get('blk_size')*self.get('N_blk')
            else:
                return self.get('N_bst')
        elif checkAnalysisType(self.AnalysisType,Jackknife):
            if self.get('use_blocking'):
                return self.get('blk_size')*self.get('N_blk')
            else:
                return self.get('data_size')
        elif checkAnalysisType(self.AnalysisType,Blocking):
            return self.get('N_blk')
        else:
            raise NotImplementedError

    def iterate_samples(self):
        # ToDo: Generalize for blocked bst/jkn
        if checkAnalysisType(self.AnalysisType,Bootstrap) or checkAnalysisType(self.AnalysisType,Jackknife):
            if self.get('use_blocking'):
                return itertools.product(range(self.get('blk_size')),range(self.get('N_blk')))

        return itertools.product(range(self.num_samples()))

    def get_sampleID(self,sample_iterator_element):
        if checkAnalysisType(self.AnalysisType,Bootstrap) or checkAnalysisType(self.AnalysisType,Jackknife):
            if self.get('use_blocking'):
                return sample_iterator_element[0] * self.get('N_blk') + sample_iterator_element[1]
        return sample_iterator_element[0]


def get_sample(t_param,t_data,t_k,t_blk_k=None):
    def _get_bst_sample(t_data):
        sample = t_data.take(t_param.bst_table[t_k,:],axis=t_param['axis'])

        return sample

    def _get_jak_sample(t_data):
        sample = np.delete( t_data,
                [t_k+i if t_k+i<t_data.shape[t_param['axis']] else i for i in range(t_param.get('N_jak'))],
                axis=t_param['axis']
        )

        return sample

    def _get_blk_sample(t_data,t_k):
        if t_k == t_param.get('N_blk') - 1:
            sample = t_data.take(range(t_k*t_param['blk_size'],t_data.shape[t_param['axis']]),axis=t_param['axis'])
        else:
            sample = t_data.take(range(t_k*t_param['blk_size'],(t_k+1)*t_param['blk_size']),axis=t_param['axis'])
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
            return _get_jak_sample(t_data= _get_blk_sample(t_data=t_data,t_k=t_blk_k))
        else:
            if t_blk_k is not None:
                warnings.warn(f"You passed 't_blk_k' = {t_blk_k} to {t_param.__name__()}.get_sample without setting 'use_blocking' to True!")

            return _get_jak_sample(t_data=t_data)

    # ==========================================================================
    # Blocking
    # ==========================================================================
    if checkAnalysisType(t_param.AnalysisType,Blocking):
            return _get_blk_sample(t_data=t_data,t_k=t_k)

def resample(t_param,t_data):
    r"""
        Return all samples in a single np.array
    """
    def _bst_resample():
        return np.array([np.mean(t_data.take(t_param.bst_table[k,:],axis=t_param['axis']),axis=t_param['axis']) for k in range(t_param['N_bst'])])

    def _jkn_resample():
        # The following removed code is suggested by Dr. Jeremy Green (Cern) and is potentially faster
        # then getting a copy of every sample.
        # However it is not so easy to generalize it for blocked jackknife
        # ToDo: Add it as a shortcut?
        #if t_param['N_jak'] == 1:
        #    # shortcut for N_jak = 1
        #    # The axis containing the data (i.e. each configuration) is given with
        #    # t_param['axis']. The np.moveaxis(x,t_param['axis'],0) create a view
        #    # on the data as if t_param['axis'] == 0.
        #    # We here return nothing else then
        #    # jak_resampled[n] = sum( data,axis=data_axis ) - data[n]
        #    return np.moveaxis(np.sum(t_data,axis=t_param['axis'])-np.moveaxis(t_data,t_param['axis'],0),0,t_param['axis'])/(t_param['data_size']-1)
        #else:
        #    # The axis containing the data (i.e. each configuration) is given with
        #    # t_param['axis']. The np.moveaxis(x,t_param['axis'],0) create a view
        #    # on the data as if t_param['axis'] == 0.
        #    # We here return nothing else then
        #    # jak_resampled[n] = sum( data,axis=data_axis ) - data[n] - data[n+1] - ... - data[n+m-1]
        #    # where m = t_param['N_jak']
        #    tmp = np.sum(t_data,axis=t_param['axis'])
        #
        #    for m in range(t_param['N_jak']):
        #        tmp -= np.moveaxis(
        #            np.moveaxis(t_data,t_param['axis'],0)[m,...],
        #            0,t_param['axis']
        #        )
        #    return tmp
        if t_param['use_blocking']:
            return np.array([
                np.mean(
                    get_sample(t_param,t_data,k_jkn,t_blk_k=k_blk), axis=t_param['axis']
                ) for k_jkn,k_blk in itertools.product(range(t_param['blk_size']),range(t_param['N_blk']))
            ])
        else:
            return np.array([
                np.mean(
                    get_sample(t_param,t_data,k_jkn), axis=t_param['axis']
                ) for k_jkn in range(t_param['data_size'])
            ])


    def _blk_resample():
        return np.array([np.mean(get_sample(t_param,t_data,k), axis=t_param['axis']) for k in range(t_param['N_blk'])])


    if checkAnalysisType(t_param.AnalysisType, Bootstrap):
        return _bst_resample()
    elif checkAnalysisType(t_param.AnalysisType, Jackknife):
        return _jkn_resample()
    else:
        return _blk_resample()


# ==============================================================================
# Jackknife
# ==============================================================================

def _jackknife_est(t_param,t_data,t_observable=None,**obs_kwargs):
    N = t_param['data_size']

    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    jkn_data = resample(t_param,obs)
    return np.mean(jkn_data, axis = t_param['axis'])

def _jackknife_var(t_param,t_data,t_observable=None,**obs_kwargs):
    N = t_param['data_size']

    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    jkn_data = resample(t_param,obs)

    return ((N-1)**2/N)*np.var(jkn_data,axis=t_param['axis'],ddof=1)

def _jackknife(t_param,t_data,t_observable=None,**obs_kwargs):
    N = t_param['data_size']

    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    jkn_data = resample(t_param,obs)

    return  np.mean( jkn_data, axis = t_param['axis'] ), \
            np.var( jkn_data, axis = t_param['axis'], ddof = 1) *((N-1)**2/N)

# ==============================================================================
# Bootstrap
# ==============================================================================

def _bootstrap_est(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.mean(bst_data,axis=0)

def _bootstrap_var(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.var(bst_data,axis=0)

def _bootstrap(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.mean(bst_data,axis=0), np.var(bst_data,axis=0)

# ==============================================================================
# Blocking
# ==============================================================================

def _blocking_est(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.mean(bst_data,axis=0)

def _blocking_var(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.var(bst_data,axis=0)

def _blocking(t_param,t_data,t_observable=None,**obs_kwargs):
    if t_observable is not None:
        obs = t_observable(t_data,**obs_kwargs)
    else:
        obs = t_data

    bst_data = resample(t_param,obs)

    return np.mean(bst_data,axis=0), np.var(bst_data,axis=0)

# ==============================================================================
# Single Interface
# ==============================================================================

def estimator(t_param,t_data,t_observable=None,**obs_kwargs):
    if checkAnalysisType(t_param.AnalysisType,Jackknife):
        return _jackknife_est(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Bootstrap):
        return _bootstrap_est(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Blocking):
        return _blocking_est(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    else:
        if t_observable is not None:
            obs = t_observable(t_data,**obs_kwargs)
        else:
            obs = t_data

        return np.mean(obs, axis=t_param['axis'])

def variance(t_param,t_data,t_observable=None,**obs_kwargs):
    if checkAnalysisType(t_param.AnalysisType,Jackknife):
        return _jackknife_var(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Bootstrap):
        return _bootstrap_var(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Blocking):
        return _blocking_var(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    else:
        if t_observable is not None:
            obs = t_observable(t_data,**obs_kwargs)
        else:
            obs = t_data

        return np.var(obs, axis=t_param['axis'])

# ==============================================================================
# Combined interface
# ==============================================================================

def dataAnalysis(t_param,t_data,t_observable=None, **obs_kwargs):
    if checkAnalysisType(t_param.AnalysisType,Jackknife):
        return _jackknife(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Bootstrap):
        return _bootstrap(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    elif checkAnalysisType(t_param.AnalysisType,Blocking):
        return _blocking(t_param,t_data,t_observable=t_observable, **obs_kwargs)
    else:
        if t_observable is not None:
            obs = t_observable(t_data,**obs_kwargs)
        else:
            obs = t_data

        return np.mean(obs,axis=t_param['axis']),np.var(obs, axis=t_param['axis'])
