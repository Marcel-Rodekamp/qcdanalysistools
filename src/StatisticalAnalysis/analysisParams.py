import numpy as np
class AnalysisParams:
    def __init__(self,t_analysis_type,t_data_size,t_with_blocking,t_num_blocks):
        # each analysis is identified with a unique string, given at parameter initialization
        self.analysis_type = t_analysis_type
        self.data_size = t_data_size
        self.with_blocking = t_with_blocking
        self.num_blocks = t_num_blocks

class JackknifeParams(AnalysisParams):
    def __init__(self,t_data_size, t_n=1, t_random_leaveout=False, t_num_ran_indices=None, t_with_blocking=False, t_num_blocks=None):
        super().__init__("jackknife",t_data_size,t_with_blocking,t_num_blocks)

        self.n = t_n

        if t_with_blocking:
            self.num_subdatasets = self.data_size // (self.n*t_num_blocks)
        else:
            self.num_subdatasets = self.data_size // (self.n)

        self.random_leaveout = t_random_leaveout

        if self.random_leaveout:
            if t_with_blocking:
                self.leave_out_index_list = np.random.randint(0,high=self.data_size // self.num_blocks-1,size=t_num_ran_indices)
            else:
                self.leave_out_index_list = np.random.randint(0,high=t_data_size-1,size=t_num_ran_indices)
                self.num_subdatasets = t_num_ran_indices
        else:
            self.leave_out_index_list = None

        self.num_ran_indices = t_num_ran_indices

class BootstrapParams(AnalysisParams):
    def __init__(self,t_data_size, t_num_subdatasets, t_with_blocking = False, t_num_blocks = None):
        super().__init__("bootstrap",t_data_size,t_with_blocking,t_num_blocks)

        self.num_subdatasets = t_num_subdatasets


class BlockingParams(AnalysisParams):
    def __init__(self,t_data_size, t_num_blocks=2):
        super().__init__("blocking",t_data_size,t_with_blocking=True,t_num_blocks=t_num_blocks)

        self.block_size = self.data_size // self.num_blocks
