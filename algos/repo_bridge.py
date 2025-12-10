import numpy as np

try:
    from approximator.routing_approximator import RoutingRmabApproximator
except Exception:
    RoutingRmabApproximator = None

from environment.routing import RoutingRMAB
from environment.scheduling import SchedulingRMAB
from environment.constrained import ConstrainedRMAB
from environment.multi_action import MultiActionRMAB
from environment.frontier_batch_env import BinaryFrontierEnvBatch   # NEW
# (or whatever the filename is where BinaryFrontierEnvBatch lives)

_solver_cache = {}


def linear_solver_approx(env):
    """
    Return a function solve(c) that maps coefficient vector c to a feasible
    binary action using the appropriate approximator for this env.
    """
    from approximator.standard_rmab_approximator import StandardRmabApproximator
    from approximator.scheduling_approximator import SchedulingRmabApproximator
    from approximator.constrained_approximator import ConstrainedRmabApproximator
    from approximator.multi_action_rmab_approximator import MultiActionRmabApproximator
    from approximator.routing_approximator import RoutingRmabApproximator
    from approximator.batch_graph_approximator import BatchGraphApproximator  # NEW

    # ---- NEW BRANCH FOR DISEASE ENV ----
    if isinstance(env, BinaryFrontierEnvBatch):
        approximator = BatchGraphApproximator(env)

    elif isinstance(env, RoutingRMAB):
        approximator = RoutingRmabApproximator(env)
    elif isinstance(env, SchedulingRMAB):
        approximator = SchedulingRmabApproximator(env)
    elif isinstance(env, ConstrainedRMAB):
        approximator = ConstrainedRmabApproximator(env)
    elif isinstance(env, MultiActionRMAB):
        approximator = MultiActionRmabApproximator(env)
    else:
        approximator = StandardRmabApproximator(env)

    def solve(c: np.ndarray) -> np.ndarray:
        return approximator.solve_from_coeffs(c)

    return solve
