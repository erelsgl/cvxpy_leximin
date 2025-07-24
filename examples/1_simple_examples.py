"""
Example: computing a leximin-egalitarian resource allocation.

See Wikipedia pages: Leximin cake-cutting, Leximin item allocation
"""

import logging
from cvxpy_leximin import Problem, Leximin, Leximax, LOGGER
from cvxpy import Variable

import numpy as np
np.set_printoptions(legacy="1.25")

print("\n## Example 1")

# There are four resources to allocate among two people: Alice and George.
# The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice:
a = Variable(4)

# The following constraint represents the fact that the allocation is feasible -- Alice gets between 0 and 1 fraction of each resource.
feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]

# Alice values the resources at 5, 3, 0, 0:
utility_Alice = a[0] * 5 + a[1] * 3 + a[2] * 0

# George values the resources at 2, 4, 7, 0:
utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 7

# A leximin allocation is an allocation that maximizes the sorted vector of utilities, in lexicographic order:
objective = Leximin([utility_Alice, utility_George])

# A problem is defined and solved just like any cvxpy problem:
problem = Problem(objective, constraints=feasible_allocation)
problem.solve()
print("Problem status: ", problem.status)
print("Objective value: ", objective.value)  # equivalent to problem.value
print(
    f"Alice's utility is {utility_Alice.value}, George's utility is {utility_George.value}."
)
# The utility vector is now ~(7.6, 7.6), which maximizes the smallest value.
print(f"The allocation is: {a.value}.")
# It is [1, 0.85, 0, 0]: Alice gets resource 0 and 85% of resource 1 (utility=7.6) and George gets 15% of resources 2 and resource 3 (utility=7.6 too).



print("\n## Example 2")
# Now, let's assume that George values the third resource at 9 (instead of 7):
utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9
problem = Problem(
    objective=Leximin([utility_Alice, utility_George]),
    constraints=feasible_allocation,
)
problem.solve()
print("Problem status: ", problem.status)
print(
    "Problem value: ", problem.value
)  # equivalent to problem.objective.value
print(
    f"Alice's utility is {utility_Alice.value}, George's utility is {utility_George.value}."
)
# The utility vector is now (8,9). It maximizes the smallest value (8), and subject to this, the next-smallest value (9).
print(f"The allocation is: {a.value}.")
# It is [1, 1, 0, 0]: Alice gets resources 0 and 1 (utility=8) and George gets resources 2 and 3 (utility=9).

print("\n## Example 2 with logging\n")
# To see the computation, activate the logger:
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)
problem.solve()

print("\n## Example 2 with a different solve method\n")
problem.solve(method="saturation")   # Default method: "ordered_outcomes"

print("\n## Example 3: leximax allocation of chores\n")
utility_Alice = a[0] * 5 + a[1] * 2 + a[2] * 0
utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9
problem = Problem(
    objective=Leximax([utility_Alice, utility_George]),
    constraints=feasible_allocation,
)
problem.solve()
print(
    f"Alice's utility is {utility_Alice.value}, George's utility is {utility_George.value}."
)
# The utility vector is now ~(2,2). It minimizes the largest value.
print(f"The allocation is: {a.value}.")
# It is ~[0, 1, 1, 0]: Alice gets resources 1 and 2 (utility=2) and George gets resources 0 and 3 (utility=2 too).



print("\n## Example 4: ordered values algorithm\n")
# To check ordered-values, we need a discrete problem, with a finite number of utility values.
# EXAMPLE 3: leximin server assignment. There are 7 identical servers to assign among 3 users.
# Each user gets some whole number of servers (0, 1, 2, 3, 4, 5, 6 or 7).
# User A values each server at 10 utility points.
# User B values each server at 15 utility points.
# User C values each server at 8 utility points.
# The variables s[0], s[1], s[2] denote the number of servers given to each user.
s = Variable(3, integer=True)
feasible_assignment = [s >= 0, sum(s) == 7, s <= 7]
utility_A = s[0] * 10
utility_B = s[1] * 15
utility_C = s[2] * 8
objective = Leximin([utility_A, utility_B, utility_C])
problem = Problem(objective, constraints=feasible_assignment)
problem.solve(method="ordered_values", outcome_levels = [105, 90, 75, 70, 60, 56, 50, 48, 45, 40, 32, 30, 24, 20, 16, 15, 10, 8, 0])
print([int(x.value) for x in s])  # servers assigned to A, B, C
print(int(utility_A.value), int(utility_B.value), int(utility_C.value))
