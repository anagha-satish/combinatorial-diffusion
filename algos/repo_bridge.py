# algos/repo_bridge.py
import numpy as np

_solver_cache = {}

def linear_solver_approx(env):
    key = id(env)
    if key not in _solver_cache:
        # Prefer the environment's advertised approximator
        Approximator = None
        if hasattr(env, "get_approximator"):
            try:
                Approximator = env.get_approximator()
            except Exception:
                Approximator = None

        # Fallbacks by class name (very light-touch)
        if Approximator is None:
            # inside linear_solver_approx(env), in the fallback name-based branch:
            name = type(env).__name__.lower()
            try:
                if "scheduling" in name:
                    from approximator.scheduling_approximator import SchedulingRmabApproximator as Approximator
                elif "constrained" in name:
                    from approximator.constrained_approximator import ConstrainedRmabApproximator as Approximator
                else:
                    from approximator.routing_approximator import RoutingRmabApproximator as Approximator
            except Exception:
                from approximator.routing_approximator import RoutingRmabApproximator as Approximator

        _solver_cache[key] = Approximator(env)

    approximator = _solver_cache[key]

    def solve(c: np.ndarray) -> np.ndarray:
        return approximator.solve_from_coeffs(c)

    return solve
