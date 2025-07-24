"""
Extend cvxpy.Problem by adding support to the Leximin and Leximax objectives.

>>> import numpy as np
>>> np.set_printoptions(legacy="1.25")  # numeric scalars are printed without their type information

EXAMPLE 1:  leximin resource allocation. There are four resources to allocate among two people.
Alice values the resources at 5, 3, 0, 0.
George values the resources at 2, 4, 9, 0.
The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice.
>>> a = cvxpy.Variable(4)
>>> feasible_allocation = [x>=0 for x in a] + [x<=1 for x in a]

>>> utility_Alice = a[0]*5 + a[1]*3 + a[2]*0
>>> utility_George   = (1-a[0])*2 + (1-a[1])*4 + (1-a[2])*9
>>> objective = Leximin([utility_Alice, utility_George])
>>> problem = Problem(objective, constraints=feasible_allocation)
>>> _=problem.solve(method="ordered_outcomes")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(8.0, 9.0)
>>> [round(x.value) for x in a]  # Alice gets all of resources 0 and 1; George gets all of resources 2 and 3.
[1, 1, 0, 0]
>>> _=problem.solve(method="saturation")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(8.0, 9.0)
>>> _=problem.solve(method="ordered_outcomes_big_M")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(8.0, 9.0)


EXAMPLE 2: leximax chores allocation. There are four chores to allocate among two people.
>>> utility_Alice = a[0]*5 + a[1]*2 + a[2]*0
>>> utility_George   = (1-a[0])*2 + (1-a[1])*4 + (1-a[2])*9
>>> objective = Leximax([utility_Alice, utility_George])
>>> problem = Problem(objective, constraints=feasible_allocation)
>>> _=problem.solve(method="ordered_outcomes")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(2.0, 2.0)
>>> [round(x.value) for x in a]  # Alice gets all of resources 1 and 2; George gets all of resources 0 and 3.
[0, 1, 1, 0]
>>> _=problem.solve(method="saturation")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(2.0, 2.0)
>>> _=problem.solve(method="ordered_outcomes_big_M")
>>> round(utility_Alice.value,1), round(utility_George.value,1)
(2.0, 2.0)


To check ordered-values, we need a discrete problem, with a finite number of utility values.
EXAMPLE 3: leximin server assignment. There are 7 identical servers to assign among 3 users.
Each user gets some whole number of servers (0, 1, 2, 3, 4, 5, 6 or 7).
User A values each server at 10 utility points.
User B values each server at 15 utility points.
User C values each server at 8 utility points.
The variables s[0], s[1], s[2] denote the number of servers given to each user.
>>> s = cvxpy.Variable(3, integer=True)
>>> feasible_assignment = [s >= 0, cvxpy.sum(s) == 7, s <= 7]

>>> utility_A = s[0] * 10
>>> utility_B = s[1] * 15
>>> utility_C = s[2] * 8
>>> objective = Leximin([utility_A, utility_B, utility_C])
>>> problem = Problem(objective, constraints=feasible_assignment)
>>> problem.solve(method="ordered_outcomes")
[20.0, 24.0, 30.0]
>>> [int(x.value) for x in s]  # servers assigned to A, B, C
[2, 2, 3]
>>> int(utility_A.value), int(utility_B.value), int(utility_C.value)
(20, 30, 24)
>>> problem.solve(method="ordered_values", outcome_levels = [105, 90, 75, 70, 60, 56, 50, 48, 45, 40, 32, 30, 24, 20, 16, 15, 10, 8, 0])
[20.0, 30.0, 24.0]
>>> [int(x.value) for x in s]  # servers assigned to A, B, C
[2, 2, 3]
>>> int(utility_A.value), int(utility_B.value), int(utility_C.value)
(20, 30, 24)

Author 1: Erel Segal-Halevi
Since: 2022-02

Author 2: Moshe Ofer
Since: 2025-06
"""

import logging
from typing import List, Optional, Union
from copy import deepcopy

import cvxpy
from cvxpy import Maximize, Minimize, error, Problem, SolverError
from cvxpy.constraints.constraint import Constraint
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.problems.problem import SizeMetrics, SolverStats

from cvxpy_leximin.objective import Leximax, Leximin

LOGGER = logging.getLogger("__cvxpy_leximin__")


def __new__init(self,
    objective: Union[Minimize, Maximize, Leximin, Leximax],
    constraints: Optional[List[Constraint]] = None,
    upper_tolerance: float = 1.01,  # for comparing floating-point numbers
    lower_tolerance: float = 0.999,
) -> None:
    if constraints is None:
        constraints = []

    # Check that objective belongs to one of the supported types
    if not isinstance(objective, (Minimize, Maximize, Leximin, Leximax)):
        raise error.DCPError("Problem objective must be Minimize, Maximize, Leximin or Leximax.")

    # Constraints and objective are immutable.
    self._objective = objective
    self._constraints = [
        cvxpy.problems.problem._validate_constraint(c) for c in constraints
    ]

    self._value = None
    self._status: Optional[str] = None
    self._solution = None
    self._cache = cvxpy.problems.problem.Cache()
    self._solver_cache = {}
    # Information about the shape of the problem and its constituent parts
    self._size_metrics: Optional["SizeMetrics"] = None
    # Benchmarks reported by the solver:
    self._solver_stats: Optional["SolverStats"] = None
    self._compilation_time: Optional[float] = None
    self._solve_time: Optional[float] = None
    self.args = [self._objective, self._constraints]
    self.upper_tolerance = upper_tolerance
    self.lower_tolerance = lower_tolerance


Problem.__init__ = __new__init


def __new__value(self):
    """float : The value/s from the last time the problem was solved."""
    if self._value is None:
        return None
    elif isinstance(self._value, list):   # For Leximin and Leximax
        return [scalar_value(v) for v in self._value]
    else:                                 # For Maximize and Minimize
        return scalar_value(self._value)


Problem.value = property(__new__value)


def _solve_sub_problem(self, sub_problem: cvxpy.Problem, *args, **kwargs):
    """
    Solves a sub-problem generated while solving a leximin / leximax problem.
    """
    kwargs = deepcopy(kwargs)  # cvxpy.Problem.solve might modify kwargs.
    sub_problem.solve(*args, **kwargs)
    if sub_problem.status == "infeasible":
        self._status = sub_problem.status
        LOGGER.warning("Sub-problem is infeasible")
        raise SolverError("Sub-problem is infeasible")
        # return False
    elif sub_problem.status == "unbounded":
        self._status = sub_problem.status
        LOGGER.warning("Sub-problem is unbounded")
        raise SolverError("Sub-problem is unbounded")
        # return False
    else:
        return True

Problem._solve_sub_problem = _solve_sub_problem

def leximin_sign(objective):
    """
    Return the arithmetic sign of the objective: +1 if it is Leximax; -1 if it is Leximin.
    """
    if type(objective) == Leximax:
        return 1
    elif type(objective) == Leximin:
        return -1
    else:
        raise SolverError(f"Unsupported objective type {type(objective)}")

def _solve_leximin_saturation(self, *args, **kwargs):
    """
    Find a leximin-optimal vector of utilities, subject to the given constraints.

    The algorithm is based on the saturation algorithm in:
    > [Stephen J. Willson](https://faculty.sites.iastate.edu/swillson/),
    > "Fair Division Using Linear Programming" (1998).
    > Part 6, pages 20--27.

    I am grateful to Sylvain Bouveret for his help with the algorithm. All remaining errors and bugs are my own.
    """
    sign = leximin_sign(self.objective)
    sub_objectives = [-sign * arg for arg in self.objective.args] # +1 for leximin, -1 for leximax

    constraints = self.constraints
    num_of_objectives = len(sub_objectives)

    # During the algorithm, the objectives are partitioned into "free" and "saturated".
    # * "free" objectives are those that can potentially be made higher, without harming the smaller objectives.
    # * "saturated" objectives are those that have already attained their highest possible value in the leximin solution.
    # Initially, all objectives are free, and no objective is saturated:
    free_objectives = list(range(num_of_objectives))
    map_saturated_objective_to_saturated_value = num_of_objectives * [None]

    while True:
        LOGGER.info("Saturated values: %s.",map_saturated_objective_to_saturated_value)
        minimum_value_for_free_objectives = cvxpy.Variable()
        inequalities_for_free_objectives = [
            sub_objectives[i] >= minimum_value_for_free_objectives
            for i in free_objectives
        ]
        inequalities_for_saturated_objectives = [
            (sub_objectives[i] >= value)
            for i, value in enumerate(map_saturated_objective_to_saturated_value)
            if value is not None
        ]

        sub_problem = cvxpy.Problem(
            objective=cvxpy.Maximize(minimum_value_for_free_objectives),
            constraints=constraints + inequalities_for_saturated_objectives + inequalities_for_free_objectives,
        )
        self._solve_sub_problem(sub_problem, *args, **kwargs)

        max_min_value_for_free_objectives = minimum_value_for_free_objectives.value.item()
        values_in_max_min_allocation = [objective.value for objective in sub_objectives]
        LOGGER.info("  max min value: %g, value-profile: %s", max_min_value_for_free_objectives, values_in_max_min_allocation)
        max_min_value_upper_threshold = (
            self.upper_tolerance * max_min_value_for_free_objectives
            if max_min_value_for_free_objectives > 0
            else (2 - self.upper_tolerance) * max_min_value_for_free_objectives
        )
        max_min_value_lower_threshold = (
            self.lower_tolerance * max_min_value_for_free_objectives
            if max_min_value_for_free_objectives > 0
            else (2 - self.lower_tolerance) * max_min_value_for_free_objectives
        )

        for ifree in free_objectives:  # Find whether i's value can be improved
            if (values_in_max_min_allocation[ifree] > max_min_value_upper_threshold):
                LOGGER.info("  Max value of objective #%d is %g, which is above %g, so objective remains free.", ifree, values_in_max_min_allocation[ifree], max_min_value_upper_threshold)
                continue
            inequalities_for_other_free_objectives = [
                sub_objectives[i] >= max_min_value_lower_threshold
                for i in free_objectives
                if i != ifree
            ]
            sub_problem_for_ifree = cvxpy.Problem(
                objective=cvxpy.Maximize(sub_objectives[ifree]),
                constraints=constraints + inequalities_for_saturated_objectives + inequalities_for_other_free_objectives,
            )
            self._solve_sub_problem(sub_problem_for_ifree, *args, **kwargs)
            max_value_for_ifree = sub_objectives[ifree].value.item()

            if max_value_for_ifree > max_min_value_upper_threshold:
                LOGGER.info("  Max value of objective #%d is %g, which is above %g, so objective remains free.", ifree, max_value_for_ifree, max_min_value_upper_threshold)
                continue
            else:
                LOGGER.info("  Max value of objective #%d is %g, which is below %g, so objective becomes saturated.", ifree, max_value_for_ifree, max_min_value_upper_threshold)
                map_saturated_objective_to_saturated_value[ifree] = max_min_value_for_free_objectives

        new_free_agents = [
            i for i in free_objectives
            if map_saturated_objective_to_saturated_value[i] is None
        ]
        if len(new_free_agents) == len(free_objectives):
            raise ValueError("No new saturated objectives - this contradicts Willson's theorem! Are you sure the domain is convex?")
        elif len(new_free_agents) == 0:
            LOGGER.info("All objectives are saturated -- values are %s.", map_saturated_objective_to_saturated_value)
            self._solve_sub_problem(sub_problem, *args, **kwargs)
            LOGGER.info("Final objective values: %s", [o.value for o in sub_objectives])

            self._status = sub_problem.status
            self._solution = sub_problem.solution

            for i in range(num_of_objectives):
                self.objective.args[i] = - sign * sub_objectives[i]
            self._value = self.objective.value
            return self.value

        else:
            free_objectives = new_free_agents
            continue


def _solve_leximin_ordered_outcomes(self, *args, **kwargs):
    """
    Solves a Leximin/Leximax problem using the Ordered Outcomes Algorithm from Ogryczak & Śliwiński (2006).

    This implementation transforms the lexicographic min-max problem into a standard lexicographic
    minimization with predefined linear criteria, avoiding integer variables.
    It creates auxiliary variables t_k and d_kj, and builds lexicographic objectives using the formula:
    lex min [t₁ + Σ d₁ⱼ, 2t₂ + Σ d₂ⱼ, ..., mt_m + Σ d_mⱼ]

    The method solves a sequence of optimization problems, each time adding constraints to
    maintain previously found optimal values.

    Reference:
    Ogryczak, W., Śliwiński, T. (2006). On Direct Methods for Lexicographic Min-Max Optimization.
    In: Gavrilova, M.L., et al. (eds) Computational Science and Its Applications - ICCSA 2006.

    Returns:
        List: Optimal values for each expression in the objective.
    """
    # The paper solves the "lex-min-max" problem. 
    # In order to solve the "lex-max-min" problem, we have to switch signs.
    sign = leximin_sign(self.objective)
    sub_objectives = [sign * arg for arg in self.objective.args]

    permanent_constraints = list(self.constraints)
    num_of_objectives = len(sub_objectives)

    # Define auxiliary variables t_k and d_kj
    t = cvxpy.Variable(num_of_objectives)
    d = cvxpy.Variable((num_of_objectives, num_of_objectives))

    # Constraints from Theorem 1
    for k in range(num_of_objectives):
        for j in range(num_of_objectives):
            permanent_constraints.append(t[k] + d[k, j] >= sub_objectives[j])
            permanent_constraints.append(d[k, j] >= 0)

    # Lexicographic optimization loop
    objectives_constraints = []
    for k in range(num_of_objectives):
        lex_obj = cvxpy.Minimize((k + 1) * t[k] + cvxpy.sum(d[k, :]))
        sub_problem = cvxpy.Problem(lex_obj, permanent_constraints + objectives_constraints)
        self._solve_sub_problem(sub_problem, *args, **kwargs) # raise SolverError on error

        # Fix the current level's value for the next level
        # fixed_value = scalar_value((k + 1) * t[k].value + sum(d[k, j].value for j in range(num_of_objectives)))
        fixed_value = scalar_value(sub_problem.value)
        LOGGER.info("Ordered outcomes iteration %s: objective value of sub-problem is %s", k, fixed_value)
        objectives_constraints.append((k + 1) * t[k] + cvxpy.sum(d[k, :]) == fixed_value)

    # Extract results
    self._value = [sign * scalar_value(t[i].value) for i in range(num_of_objectives)]
    self._status = sub_problem.status
    self._solution = sub_problem.solution  # The solution of the last sub_problem is the solution of the entire problem

    LOGGER.info("Final objective values: %s", self._value)

    return self._value


def _solve_leximin_ordered_outcomes_big_M(self, big_M=1e5, *args, **kwargs):
    """
    Solves a Leximin/Leximax problem using the Ordered Outcomes Algorithm from Ogryczak & Śliwiński (2006).

    This implementation transforms the lexicographic min-max problem into a standard lexicographic
    minimization with predefined linear criteria, avoiding integer variables.
    It creates auxiliary variables t_k and d_kj, and builds lexicographic objectives using the formula:
    lex min [t₁ + Σ d₁ⱼ, 2t₂ + Σ d₂ⱼ, ..., mt_m + Σ d_mⱼ]

    The method solves a sequence of optimization problems, each time adding constraints to
    maintain previously found optimal values.

    Reference:
    Ogryczak, W., Śliwiński, T. (2006). On Direct Methods for Lexicographic Min-Max Optimization.
    In: Gavrilova, M.L., et al. (eds) Computational Science and Its Applications - ICCSA 2006.

    Returns:
        List: Optimal values for each expression in the objective.
    """
    sign = leximin_sign(self.objective)
    sub_objectives = [sign * arg for arg in self.objective.args]

    constraints = list(self.constraints)
    num_of_objectives = len(sub_objectives)

    # Decision variables for ordered outcomes
    t = cvxpy.Variable(num_of_objectives)
    z = cvxpy.Variable((num_of_objectives, num_of_objectives), boolean=True)

    # Enforce constraints from the Ordered Outcomes model
    for k in range(num_of_objectives):
        # Each t_k must be greater than or equal to all but (k) outcomes
        for j in range(num_of_objectives):
            constraints.append(t[k] - sub_objectives[j] >= -big_M * z[k, j])
        constraints.append(cvxpy.sum(z[k, :]) <= k)

    # Lexicographic optimization — sequentially solve for (t_0, t_1, ..., t_{num_of_objectives-1})
    objectives = []
    for k in range(num_of_objectives):
        obj = cvxpy.Minimize(t[k])
        sub_problem = cvxpy.Problem(obj, constraints + objectives)
        sub_problem.solve()

        if sub_problem.status != cvxpy.OPTIMAL:
            raise RuntimeError(f"Subproblem {k} not solved to optimality.")

        self._status = sub_problem.status
        # Fix the previous value for next level of lex order
        objectives.append(t[k] == t[k].value)

    # Store final values
    self._value = [sign * scalar_value(ti.value) for ti in t]
    self._solution = sub_problem.solution

    return self._value


def _solve_leximin_ordered_values(self, *args, **kwargs):
    """
    Solve the Leximin or Leximax problem using the Ordered Values algorithm
    (Theorem 3, formulation (7)) from Ogryczak & Śliwiński (2006).

    Parameters:
        outcome_levels (List[float], optional): sorted list in descending order of all possible distinct outcome values.
            If None, it must be known or estimated elsewhere in the problem.

    Returns:
        List[float]: The optimal lexicographic objective values.
    """
    outcome_levels = kwargs.pop("outcome_levels", None)
    if outcome_levels is None:
        raise ValueError("The list of possible outcome values (outcome_levels) must be provided.")

    sign = leximin_sign(self.objective)
    sub_objectives = [sign * arg for arg in self.objective.args]
    
    # Apply the same sign transformation to outcome_levels
    transformed_outcome_levels = sorted([sign * level for level in outcome_levels], reverse=True)

    constraints = list(self.constraints)
    num_of_objectives = len(sub_objectives)

    r = len(transformed_outcome_levels)

    # Variables: h_{kj} for k = 2..r and j = 1..num_of_objectives
    h = {
        (k, j): cvxpy.Variable(name=f"h_{k}_{j}", nonneg=True)
        for k in range(1, r)
        for j in range(num_of_objectives)
    }

    # Constraints for h_{kj} ≥ f_j(x) - v_k
    for k in range(1, r):
        v_k = transformed_outcome_levels[k]
        for j in range(num_of_objectives):
            constraints.append(h[(k, j)] >= sub_objectives[j] - v_k)

    # Lexicographic minimization: minimize sum_j h_{kj} in order of k
    objectives = []
    for k in range(1, r):
        obj_k = cvxpy.Minimize(cvxpy.sum([h[(k, j)] for j in range(num_of_objectives)]))
        problem_k = cvxpy.Problem(obj_k, constraints + objectives)
        problem_k.solve()

        if problem_k.status != cvxpy.OPTIMAL:
            raise RuntimeError(f"Subproblem {k} not solved to optimality.")

        # Fix the value for the current step
        fixed_val = scalar_value(sum(h[(k, j)].value for j in range(num_of_objectives)))
        LOGGER.info("Ordered values iteration k=%s: sum(h[k,j]) = %s", k, fixed_val)
        objectives.append(cvxpy.sum([h[(k, j)] for j in range(num_of_objectives)]) == fixed_val)

    self._value = self.objective.value
    self._status = problem_k.status
    self._solution = problem_k.solution

    return self._value


# Choose default leximin algorithm here
Problem._solve_leximin = _solve_leximin_ordered_outcomes
# Problem._solve_leximin = _solve_leximin_saturation


def __new__solve(self, *args, **kwargs):
    # print("__new_solve args: ",args)
    func_name = kwargs.pop("method", None)
    if func_name is not None:
        solve_func = Problem.REGISTERED_SOLVE_METHODS[func_name]
    else:
        if type(self.objective) in [Leximin,Leximax]:
            solve_func = Problem._solve_leximin
        else:
            solve_func = Problem._solve
    # LOGGER.debug("solve function = '%s'", solve_func.__name__)
    return solve_func(self, *args, **kwargs)


Problem.register_solve("willson", _solve_leximin_saturation)
Problem.register_solve("saturation", _solve_leximin_saturation)

Problem.register_solve("ogry_relax", _solve_leximin_ordered_outcomes)
Problem.register_solve("ordered_outcomes", _solve_leximin_ordered_outcomes)

Problem.register_solve("ogry_integer", _solve_leximin_ordered_outcomes_big_M)
Problem.register_solve("ordered_outcomes_big_M", _solve_leximin_ordered_outcomes_big_M)

Problem.register_solve("ordered_values", _solve_leximin_ordered_values)

Problem.solve = __new__solve

if __name__ == "__main__":
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.setLevel(logging.INFO)

    import doctest
    print(doctest.testmod())