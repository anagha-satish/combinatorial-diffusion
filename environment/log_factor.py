# Standard library imports

# Third-party imports
import numpy as np
from scipy.special import logsumexp

# Local imports

class LogFactor:
    """
    A factor in log-domain over discrete variables
    vars  : list of variable names (e.g. ['A','B'])
    table : numpy array of shape (card(A), card(B), ...) already in log-space
    """
    def __init__(self, vars, table, is_log=False):
        self.vars = list(vars)
        arr = np.array(table)
        self.table = arr if is_log else np.log(arr)

    def reduce(self, var, val):
        """
        Condition on var = val, slicing out that dimension
        """
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        slicer = [slice(None)] * len(self.vars)
        slicer[i] = val
        new_tbl = self.table[tuple(slicer)]
        new_vars = self.vars[:i] + self.vars[i+1:]
        return LogFactor(new_vars, new_tbl, is_log=True)

    def multiply(self, other, domains):
        """
        Multiply two log-factors then add their tables over a broadcast grid
        domains: dict var -> cardinality
        """
        new_vars = sorted(set(self.vars + other.vars))
        
        # Prepare shapes for broadcasting
        def reshape_for(f):
            shape = [(domains[v] if v in f.vars else 1) for v in new_vars]
            return f.table.reshape(shape)

        A = reshape_for(self)
        B = reshape_for(other)
        return LogFactor(new_vars, A + B, is_log=True)

    def marginalize(self, var):
        """
        Sum-out (log-sum-exp) variable var
        """
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        new_tbl = logsumexp(self.table, axis=i)
        new_vars = self.vars[:i] + self.vars[i+1:]
        return LogFactor(new_vars, new_tbl, is_log=True)
