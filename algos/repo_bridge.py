import numpy as np

try:
    from approximator.routing_approximator import RoutingRmabApproximator
except Exception:
    RoutingRmabApproximator = None

_solver_cache = {} 

def linear_solver_approx(env):
    from approximator.routing_approximator import RoutingRmabApproximator
    approximator = RoutingRmabApproximator(env)

    def solve(c: np.ndarray) -> np.ndarray:
        return approximator.solve_from_coeffs(c)

    return solve