import numpy as np
from typing import Dict, List, Tuple
import warnings

from ..optimize import Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem
from .solver import InnerSolver
from .optimal_scaling_problem import OptimalScalingProblem

REDUCED = 'reduced'
STANDARD = 'standard'
MAXMIN = 'max-min'
MAX = 'max'


class OptimalScalingInnerSolver(InnerSolver):
    """
    Solve the inner subproblem of the
    optimal scaling approach for ordinal data.
    """

    def __init__(self,
                 optimizer: Optimizer = None,
                 options: Dict = None):

        self.optimizer = optimizer
        self.options = options
        if self.options is None:
            self.options = OptimalScalingInnerSolver.get_default_options()
        if self.options['method'] == STANDARD \
                and self.options['reparameterized']:
            raise NotImplementedError(
                'Combining standard approach with '
                'reparameterization not implemented.'
            )
        self.x_guesses = None

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> list:
        """
        Get results for every group (inner optimization problem)

        Parameters
        ----------
        problem:
            InnerProblem from pyPESTO hierarchical
        sim:
            Simulations from AMICI
        sigma:
            List of sigmas (not needed for this approach)
        scaled:
            ...
        """
        optimal_surrogates = []
        for gr in problem.get_groups_for_xs(InnerParameter.OPTIMALSCALING):
            xs = problem.get_xs_for_group(gr)
            surrogate_opt_results = optimize_surrogate_data(xs, sim, self.options)
            optimal_surrogates.append(surrogate_opt_results)
        return optimal_surrogates

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """
        Calculate the inner objective function from a list of inner
        optimization results returned from compute_optimal_surrogate_data

        Parameters
        ----------
        x_inner_opt:
            List of optimization results
        """

        if False in [x_inner_opt[idx]['success'] for idx in range(len(x_inner_opt))]:
            obj = np.nan
            warnings.warn(f"Inner optimization failed.")
        else:
            obj = np.sum(
                [x_inner_opt[idx]['fun'] for idx in range(len(x_inner_opt))]
            )
        return obj

    def calculate_gradients(self, problem, x_inner_opt, sim, sy):
        grad = 0.0
        par_idx = 0
        for idx, gr in enumerate(problem.get_groups_for_xs(InnerParameter.OPTIMALSCALING)):
            xi = get_xi(gr, problem, x_inner_opt[idx], sim, self.options)
            sim_all = get_sim_all(problem.get_xs_for_group(gr), sim)
            sy_all = get_sy_all(problem.get_xs_for_group(gr), sim, sy, par_idx)
            res = np.block([xi[:problem.num_datapoints] - sim_all,
                            np.zeros(problem.num_inner_params - problem.num_datapoints)])

            df_dxi = 2 * problem.W.dot(res)

            mu = get_mu(problem, xi)

            # for theta_i in theta:
            dy_dtheta = get_dy_dtheta(problem, sy)

            dxi_dtheta = calculate_dxi_dtheta(problem, xi, mu, dy_dtheta)

            df_dtheta = -2 * problem.W.dot(dy_dtheta).dot(res)

            grad += dxi_dtheta.dot(df_dxi) + df_dtheta
        return grad

    @staticmethod
    def get_default_options() -> Dict:
        """
        Return default options for solving the inner problem,
        if no options provided
         """
        options = {'method': 'reduced',
                   'reparameterized': True,
                   'intervalConstraints': 'max',
                   'minGap': 1e-16}
        return options


def calculate_dxi_dtheta(problem, xi, mu, dy_dtheta):
    from scipy import linalg
    A = np.block([[2 * problem.W, problem.C.transpose()],
                  [(mu*problem.C.transpose()).transpose(), np.diag(problem.C.dot(xi))]])

    b = np.block([2*dy_dtheta.dot(problem.W), np.zeros(problem.num_constr_full)])

    dxi_dtheta = linalg.lstsq(A, b)
    return dxi_dtheta[0][:problem.num_inner_params]


def get_dy_dtheta(problem, sy):
    dy_dtheta = np.zeros(problem.num_inner_params)
    # TODO: wrong order of datapoints
    dy_dtheta[:problem.num_datapoints] = np.array([sy[idx][0][5][0] for idx in range(len(sy))])
    dy_dtheta = dy_dtheta[[1, 0, 2]]

    return np.block([dy_dtheta, np.zeros(2*problem.num_categories)])


def get_mu(problem: OptimalScalingProblem, xi):
    mu = np.zeros(problem.num_constr_full)
    for idx in range(problem.num_datapoints):
        cat_idx = problem.get_cat_for_xi_idx(idx)
        x_lower = xi[problem.lb_indices[cat_idx]]
        x_upper = xi[problem.ub_indices[cat_idx]]
        y_surr = xi[idx]
        if np.isclose(y_surr, x_lower):
            mu[idx] = 1
        if np.isclose(y_surr, x_upper):
            mu[problem.num_datapoints + idx] = 1

    for idx in range(problem.num_categories - 1):
        x_lower = xi[problem.lb_indices[idx] + 1]
        x_upper = xi[problem.ub_indices[idx]]
        if np.isclose(x_lower, x_upper):
            mu[2*problem.num_datapoints + idx] = 1

    return mu


def get_xi(gr,
           problem: OptimalScalingProblem,
           x_inner_opt: Dict,
           sim: List[np.ndarray],
           options: Dict):

    xs = problem.get_xs_for_group(gr)
    interval_range, interval_gap = \
        compute_interval_constraints(xs, sim, options)

    xi = np.zeros(problem.num_inner_params)
    surrogate_all, x_lower, x_upper = \
        get_surrogate_all(xs, x_inner_opt['x'], sim, interval_range, interval_gap, options)
    xi[:problem.num_datapoints] = surrogate_all.flatten()
    xi[problem.lb_indices] = x_lower
    xi[problem.ub_indices] = x_upper
    return xi


def optimize_surrogate_data(xs: List[InnerParameter],
                            sim: List[np.ndarray],
                            options: Dict):
    """Run optimization for inner problem"""

    from scipy.optimize import minimize

    interval_range, interval_gap = \
        compute_interval_constraints(xs, sim, options)
    w = get_weight_for_surrogate(xs, sim)

    def obj_surr(x):
        return obj_surrogate_data(xs, x, sim, interval_gap,
                                  interval_range, w, options)

    inner_options = \
        get_inner_options(options, xs, sim, interval_range, interval_gap)

    results = minimize(obj_surr, **inner_options)
    return results


def get_inner_options(options: Dict,
                      xs: List[InnerParameter],
                      sim: List[np.ndarray],
                      interval_range: float,
                      interval_gap: float) -> Dict:

    """Return default options for scipy optimizer"""

    from scipy.optimize import Bounds

    min_all, max_all = get_min_max(xs, sim)
    if options['method'] == REDUCED:
        parameter_length = len(xs)
        x0 = np.linspace(
            np.max([min_all, interval_range]),
            max_all + interval_range,
            parameter_length
        )
    elif options['method'] == STANDARD:
        parameter_length = 2 * len(xs)
        x0 = np.linspace(0, max_all + interval_range, parameter_length)
    else:
        raise NotImplementedError(
            f"Unkown optimal scaling method {options['method']}. "
            f"Please use {STANDARD} or {REDUCED}."

        )

    if options['reparameterized']:
        x0 = y2xi(x0, xs, interval_gap, interval_range)
        bounds = Bounds([0] * parameter_length, [max_all] * parameter_length)

        inner_options = {'x0': x0, 'method': 'L-BFGS-B',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'bounds': bounds}
    else:
        constraints = get_constraints_for_optimization(xs, sim, options)

        inner_options = {'x0': x0, 'method': 'SLSQP',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'constraints': constraints}
    return inner_options


def get_min_max(xs: List[InnerParameter],
                sim: List[np.ndarray]) -> Tuple[float, float]:
    """Return minimal and maximal simulation value"""

    sim_all = get_sim_all(xs, sim)

    min_all = np.min(sim_all)
    max_all = np.max(sim_all)

    return min_all, max_all


def get_sy_all(xs, sim, sy, par_idx):
    sy_all = []
    for x in xs:
        for sy_i, mask_i in \
                zip(sy, x.ixs):
            sim_sy = sy_i[mask_i]
            if mask_i.any():
                sy_all.append(sim_sy[0])
    return 0


def get_sim_all(xs, sim: List[np.ndarray]) -> list:
    """"Get list of all simulations for all xs"""

    sim_all = []
    for x in xs:
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            sim_x = sim_i[mask_i]
            if mask_i.any():
                for sim_x_i in sim_x:
                    sim_all.append(sim_x_i)
    return sim_all


def get_surrogate_all(xs,
                      optimal_scaling_bounds,
                      sim,
                      interval_range,
                      interval_gap,
                      options):
    if options['reparameterized']:
        optimal_scaling_bounds = \
            xi2y(optimal_scaling_bounds, xs, interval_gap, interval_range)
    surrogate_all = []
    x_lower_all = []
    x_upper_all = []
    for x in xs:
        x_upper, x_lower = \
            get_bounds_for_category(
                x, optimal_scaling_bounds, interval_gap, options
            )
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            if mask_i.any():
                y_sim = sim_i[mask_i]
                for y_sim_i in y_sim:
                    if x_lower > y_sim_i:
                        y_surrogate = x_lower
                    elif y_sim_i > x_upper:
                        y_surrogate = x_upper
                    else:
                        y_surrogate = y_sim_i
                    surrogate_all.append(y_surrogate)
                    if x_lower not in x_lower_all:
                        x_lower_all.append(x_lower)
                    if x_upper not in x_upper_all:
                        x_upper_all.append(x_upper)
    return np.array(surrogate_all), np.array(x_lower_all), np.array(x_upper_all)


def get_weight_for_surrogate(xs: List[InnerParameter],
                             sim: List[np.ndarray]) -> float:
    """Calculate weights for objective function"""

    sim_x_all = get_sim_all(xs, sim)
    eps = 1e-10
    v_net = 0
    for idx in range(len(sim_x_all) - 1):
        v_net += np.abs(sim_x_all[idx + 1] - sim_x_all[idx])
    w = 0.5 * np.sum(np.abs(sim_x_all)) + v_net + eps
    return 1  # TODO: w ** 2


def compute_interval_constraints(xs: List[InnerParameter],
                                 sim: List[np.ndarray],
                                 options: Dict) -> Tuple[float, float]:
    """Compute minimal interval range and gap"""

    # compute constraints on interval size and interval gap size
    # similar to Pargett et al. (2014)
    if 'minGap' not in options:
        eps = 1e-16
    else:
        eps = options['minGap']

    min_simulation, max_simulation = get_min_max(xs, sim)

    if options['intervalConstraints'] == MAXMIN:

        interval_range = \
            (max_simulation - min_simulation) / (2 * len(xs) + 1)
        interval_gap = \
            (max_simulation - min_simulation) / (4 * (len(xs) - 1) + 1)
    elif options['intervalConstraints'] == MAX:

        interval_range = max_simulation / (2 * len(xs) + 1)
        interval_gap = max_simulation / (4 * (len(xs) - 1) + 1)
    else:
        raise ValueError(
            f"intervalConstraints = "
            f"{options['intervalConstraints']} not implemented. "
            f"Please use {MAX} or {MAXMIN}."

        )
    if interval_gap < eps:
        interval_gap = eps
    return 0.0, 0.0  # TODO: interval_range, interval_gap


def y2xi(optimal_scaling_bounds: np.ndarray,
         xs: List[InnerParameter],
         interval_gap: float,
         interval_range: float) -> np.ndarray:
    """Get optimal scaling bounds and return reparameterized parameters"""

    optimal_scaling_bounds_reparameterized = \
        np.full(shape=(np.shape(optimal_scaling_bounds)), fill_value=np.nan)

    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            optimal_scaling_bounds_reparameterized[x_category - 1] = \
                optimal_scaling_bounds[x_category - 1] \
                - interval_range
        else:
            optimal_scaling_bounds_reparameterized[x_category - 1] = \
                optimal_scaling_bounds[x_category - 1] \
                - optimal_scaling_bounds[x_category - 2] \
                - interval_gap - interval_range

    return optimal_scaling_bounds_reparameterized


def xi2y(
        optimal_scaling_bounds_reparameterized: np.ndarray,
        xs: List[InnerParameter],
        interval_gap: float,
        interval_range: float) -> np.ndarray:
    """
    Get reparameterized parameters and
    return original optimal scaling bounds
    """

    # TODO: optimal scaling parameters in
    #  parameter sheet have to be ordered at the moment
    optimal_scaling_bounds = \
        np.full(shape=(np.shape(optimal_scaling_bounds_reparameterized)),
                fill_value=np.nan)
    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            optimal_scaling_bounds[x_category - 1] = \
                interval_range + optimal_scaling_bounds_reparameterized[
                    x_category - 1]
        else:
            optimal_scaling_bounds[x_category - 1] = \
                optimal_scaling_bounds_reparameterized[x_category - 1] + \
                interval_gap + interval_range + optimal_scaling_bounds[
                    x_category - 2]
    return optimal_scaling_bounds


def obj_surrogate_data(xs: List[InnerParameter],
                       optimal_scaling_bounds: np.ndarray,
                       sim: List[np.ndarray],
                       interval_gap: float,
                       interval_range: float,
                       w: float,
                       options: Dict) -> float:
    """compute optimal scaling objective function"""

    obj = 0.0
    if options['reparameterized']:
        optimal_scaling_bounds = \
            xi2y(optimal_scaling_bounds, xs, interval_gap, interval_range)

    for x in xs:
        x_upper, x_lower = \
            get_bounds_for_category(
                x, optimal_scaling_bounds, interval_gap, options
            )
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            if mask_i.any():
                y_sim = sim_i[mask_i]
                for y_sim_i in y_sim:
                    if x_lower > y_sim_i:
                        y_surrogate = x_lower
                    elif y_sim_i > x_upper:
                        y_surrogate = x_upper
                    else:
                        y_surrogate = y_sim_i
                    obj += (y_surrogate - y_sim_i) ** 2
    obj = np.divide(obj, w)
    return obj


def get_bounds_for_category(x: InnerParameter,
                            optimal_scaling_bounds: np.ndarray,
                            interval_gap: float,
                            options: Dict) -> Tuple[float, float]:
    """Return upper and lower bound for a specific category x"""

    x_category = int(x.category)

    if options['method'] == REDUCED:
        x_upper = optimal_scaling_bounds[x_category - 1]
        if x_category == 1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        else:
            raise ValueError('Category value needs to be larger than 0.')
    elif options['method'] == STANDARD:
        x_lower = optimal_scaling_bounds[2 * x_category - 2]
        x_upper = optimal_scaling_bounds[2 * x_category - 1]
    else:
        raise NotImplementedError(
            f"Unkown optimal scaling method {options['method']}. "
            f"Please use {REDUCED} or {STANDARD}."

        )
    return x_upper, x_lower


def get_constraints_for_optimization(xs: List[InnerParameter],
                                     sim: List[np.ndarray],
                                     options: Dict) -> Dict:
    """Return constraints for inner optimization"""

    num_categories = len(xs)
    interval_range, interval_gap = \
        compute_interval_constraints(xs, sim, options)
    if options['method'] == REDUCED:
        a = np.diag(-np.ones(num_categories), -1) \
            + np.diag(np.ones(num_categories + 1))
        a = a[:-1, :-1]
        b = np.empty((num_categories,))
        b[0] = interval_range
        b[1:] = interval_range + interval_gap
    elif options['method'] == STANDARD:
        a = np.diag(-np.ones(2 * num_categories), -1) \
            + np.diag(np.ones(2 * num_categories + 1))
        a = a[:-1, :]
        a = a[:, :-1]
        b = np.empty((2 * num_categories,))
        b[0] = 0
        b[1::2] = interval_range
        b[2::2] = interval_gap
    ineq_cons = {'type': 'ineq', 'fun': lambda x: a.dot(x) - b}

    return ineq_cons