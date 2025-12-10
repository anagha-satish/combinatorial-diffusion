# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Local imports

class AbstractJointProbabilityClass(ABC):
    def __init__(self, variables: list[str], args: dict) -> None:
        self.variables = variables
        self.n = len(variables)
        self.args = args
        self.unary_factors = dict()
        self.pairwise_factors = dict()
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_conditional_probability(self, query_dict: dict, evidence_dict: dict) -> float:
        raise NotImplementedError

    @staticmethod
    def compute_theta_length(covariate_length: int, order: int) -> int:
        if order == 1:
            return 2 + 2 * covariate_length
        elif order == 2:
            return 4 + 5 * covariate_length
        else:
            assert False

    @staticmethod
    def f_unary(x_i: int, covariates_i: np.ndarray) -> np.ndarray:
        assert x_i == 0 or x_i == 1
        output = [1, x_i]
        for idx in range(len(covariates_i)):
            output += [
                covariates_i[idx],
                x_i * covariates_i[idx]
            ]
        assert len(output) == 2 + 2 * len(covariates_i)
        return np.array(output)

    @staticmethod
    def f_pairwise(x_i: int, x_j: int, covariates_i: np.ndarray, covariates_j: np.ndarray) -> np.ndarray:
        assert (x_i == 0 or x_i == 1) and (x_j == 0 or x_j == 1)
        assert len(covariates_i) == len(covariates_j)
        xixj = x_i * x_j
        nxixj = (1 - x_i) * x_j
        xinxj = x_i * (1 - x_j)
        nxinxj = (1 - x_i) * (1 - x_j)

        output = [1, xixj, nxixj + xinxj, nxinxj]
        for idx in range(len(covariates_i)):
            output += [
                covariates_i[idx] + covariates_j[idx],
                xixj * (covariates_i[idx] + covariates_j[idx]),
                xinxj * covariates_i[idx] + nxixj * covariates_j[idx],
                nxixj * covariates_i[idx] + xinxj * covariates_j[idx],
                nxinxj * (covariates_i[idx] + covariates_j[idx]),
            ]
        assert len(output) == 4 + 5 * len(covariates_i)
        return np.array(output)
    
    @staticmethod
    def logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
        m = np.max(x, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)