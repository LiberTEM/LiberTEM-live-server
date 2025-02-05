import numpy as np

from libertem.udf.sumsigudf import SumSigUDF

class SumSigSegUDF(SumSigUDF):
    """
    Sum over the signal axes. For each navigation position, the sum of all pixels is calculated.

    Examples
    --------
    >>> udf = SumSigUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["intensity"]).shape
    (16, 16)
    """
    

    def __init__(self, N_x = 1, N_y = 1, *args, **kwargs):
        """
        Parameters
        ----------
        """
        super().__init__(*args, N_x=N_x, N_y=N_y, **kwargs)

    def get_result_buffers(self):
        """"""
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        results = {}
        for i in range (self.params.N_x):
            results[f"intensity{i}"] = self.buffer(
                kind="single", dtype=dtype, where="device", 
                extra_shape= 
                (self.meta.dataset_shape[0] // self.params.N_y,
                    self.meta.dataset_shape[1] // self.params.N_x)
            )
        results["intensity"] = self.buffer(kind="nav", dtype=dtype, where="device")

        return results
    
    def get_results(self):
        results = {}
        allintens = self.results.get_buffer("intensity").data
        results["intensity"] = allintens
        for i in range(self.params.N_x):
            buff = self.results.get_buffer(f"intensity{i}").data
            subdata = allintens[:,i::self.params.N_x].copy()
            nonzero = subdata>0
            if np.any(nonzero):           
                subdata[~nonzero] = np.mean(subdata[nonzero])
            buff[:] = subdata[:buff.shape[0], :buff.shape[1]]
            results[f"intensity{i}"] = buff

        return results
    
    def merge(self, dest, src):
        dest.intensity += src.intensity