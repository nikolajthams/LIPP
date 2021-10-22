import numpy as np
from patsy import dmatrix, build_design_matrices

class Basis():
    """Object transforming data by basis expansion to design matrix."""
    def __init__(self, t_span, df, intercept=False):
        design = f"bs(x, df = {df})" if intercept else f"bs(x, df = {df})-1"
        self.d = dmatrix(design, {"x": np.array([*t_span])})
        self.span = t_span
        self.ncols = self.d.shape[1]

    def transform(self, data, cut_0 = False):
        if data.size == 0: return np.empty((0, self.ncols))
        if not cut_0:
            return np.asarray(build_design_matrices([self.d.design_info], {"x" : data})[0])
        else:
            # We expect values outside t_span. Find which are within span, and only transform those
            out = np.zeros((len(data), self.d.shape[1]))
            valid = np.where((self.span[0] <= data) & (data <= self.span[1]))
            out[valid] = np.asarray(build_design_matrices([self.d.design_info], {"x" : data[valid]})[0])
            return out
