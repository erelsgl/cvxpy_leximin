"""
Extend cvxpy.Problem by adding support to the Leximin and Leximax objectives.

Author: Erel Segal-Halevi
Since: 2022-02
"""

import cvxpy

from cvxpy import Minimize, Maximize, error
from cvxpy.constraints.constraint import Constraint
from cvxpy_leximin.objective import Leximin, Leximax
import cvxpy.utilities as u
from typing import Dict, List, Optional, Union
from solve import solve
import logging

logger = logging.getLogger(__name__)


class Problem(cvxpy.problems.problem.Problem):
    def __init__(
        self,
        objective: Union[Minimize, Maximize, Leximin, Leximax],
        constraints: Optional[List[Constraint]] = None,
        upper_tolerance: float = 1.01,  # for comparing floating-point numbers
        lower_tolerance: float = 0.999,
    ) -> None:
        if constraints is None:
            constraints = []
        # Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize, Leximin, Leximax)):
            raise error.DCPError("Problem objective must be Minimize, Maximize, Leximin or Leximax.")
        # Constraints and objective are immutable.
        self._objective = objective
        self._constraints = [cvxpy.problems.problem._validate_constraint(c) for c in constraints]

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

    def _solve_leximin(self, objectives: list, constraints: list, *args, **kwargs):
        """
        Find a leximin-optimal vector of utilities, subject to the given constraints.

        :param objectives: A list of cvxpy expressions, representing the various objectives. The order in the list is irrelevant.
        :param constraints: A list of cvxpy constraints. The constraints must specify a convex domain.
        :param kwargs: keyword arguments passed on to cvxpy.Problem.solve().
        :return None. When the function completes, you can access the values of the variables and objectives in the leximin solution using the Variable.value field.

        EXAMPLE: resource allocation. There are three resources to allocate among two people.
        Alice values the resources at 5, 3, 0.
        Bob values the resources at 2, 4, 9.
        The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice.
        >>> a = cvxpy.Variable(4)
        >>> feasible_allocation = [x>=0 for x in a] + [x<=1 for x in a]

        >>> utility_Alice = a[0]*5 + a[1]*3 + a[2]*0
        >>> utility_Bob   = (1-a[0])*2 + (1-a[1])*4 + (1-a[2])*9
        >>> objective = Leximin([utility_Alice, utility_Bob])
        >>> problem = Problem(objective, constraints=feasible_allocation)
        >>> problem.solve()
        >>> round(utility_Alice.value), round(utility_Bob.value)
        (8, 9)
        >>> [round(x.value) for x in a]  # Alice gets all of resources 0 and 1; Bob gets all of resources 2 and 3.
        [1, 1, 0, 0]
        """
        num_of_objectives = len(objectives)

        # During the algorithm, the objectives are partitioned into "free" and "saturated".
        # * "free" objectives are those that can potentially be made higher, without harming the smaller objectives.
        # * "saturated" objectives are those that have already attained their highest possible value in the leximin solution.
        # Initially, all objectives are free, and no objective is saturated:
        free_objectives = list(range(num_of_objectives))
        map_saturated_objective_to_saturated_value = num_of_objectives * [None]

        while True:
            logger.info("Saturated values: %s.", map_saturated_objective_to_saturated_value)
            minimum_value_for_free_objectives = cvxpy.Variable()
            inequalities_for_free_objectives = [
                objectives[i] >= minimum_value_for_free_objectives for i in free_objectives
            ]
            inequalities_for_saturated_objectives = [
                (objectives[i] >= value)
                for i, value in enumerate(map_saturated_objective_to_saturated_value)
                if value is not None
            ]

            problem = cvxpy.Problem(
                objective=cvxpy.Maximize(minimum_value_for_free_objectives),
                constraints=constraints + inequalities_for_saturated_objectives + inequalities_for_free_objectives,
            )
            solve(problem, **kwargs)  # , solvers=solvers
            max_min_value_for_free_objectives = minimum_value_for_free_objectives.value.item()

            values_in_max_min_allocation = [objective.value for objective in objectives]
            logger.info(
                "  max min value: %g, value-profile: %s",
                max_min_value_for_free_objectives,
                values_in_max_min_allocation,
            )
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
                if values_in_max_min_allocation[ifree] > max_min_value_upper_threshold:
                    logger.info(
                        "  Max value of objective #%d is %g, which is above %g, so objective remains free.",
                        ifree,
                        values_in_max_min_allocation[ifree],
                        max_min_value_upper_threshold,
                    )
                    continue
                inequalities_for_other_free_objectives = [
                    objectives[i] >= max_min_value_lower_threshold for i in free_objectives if i != ifree
                ]

                problem = cvxpy.Problem(
                    objective=cvxpy.Maximize(objectives[ifree]),
                    constraints=constraints
                    + inequalities_for_saturated_objectives
                    + inequalities_for_other_free_objectives,
                )
                solve(problem, **kwargs)  # , solvers=solvers
                max_value_for_ifree = objectives[ifree].value.item()

                if max_value_for_ifree > max_min_value_upper_threshold:
                    logger.info(
                        "  Max value of objective #%d is %g, which is above %g, so objective remains free.",
                        ifree,
                        max_value_for_ifree,
                        max_min_value_upper_threshold,
                    )
                    continue
                else:
                    logger.info(
                        "  Max value of objective #%d is %g, which is below %g, so objective becomes saturated.",
                        ifree,
                        max_value_for_ifree,
                        max_min_value_upper_threshold,
                    )
                    map_saturated_objective_to_saturated_value[ifree] = max_min_value_for_free_objectives

            new_free_agents = [i for i in free_objectives if map_saturated_objective_to_saturated_value[i] is None]
            if len(new_free_agents) == len(free_objectives):
                raise ValueError(
                    "No new saturated objectives - this contradicts Willson's theorem! Are you sure the domain is convex?"
                )
            elif len(new_free_agents) == 0:
                logger.info(
                    "All objectives are saturated -- values are %s.",
                    map_saturated_objective_to_saturated_value,
                )
                return
            else:
                free_objectives = new_free_agents
                continue

    def solve(self, *args, **kwargs):
        if type(self.objective) == Maximize or type(self.objective) == Minimize:
            return super().solve(*args, **kwargs)
        elif type(self.objective) == Leximin:
            return self._solve_leximin(self.objective.args, self.constraints, *args, **kwargs)
        else:
            raise NotImplementedError(f"Objective of type {type(self.objective)} is not supported")


Problem.logger = logger


if __name__ == "__main__":
    import sys

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    import doctest

    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))
