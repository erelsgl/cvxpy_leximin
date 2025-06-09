"""
Example: computing a leximin-egalitarian resource allocation.

See Wikipedia pages: Leximin cake-cutting, Leximin item allocation
"""

import cvxpy, logging
from cvxpy_leximin import Problem, Leximin, Leximax, LOGGER

import numpy as np
np.set_printoptions(legacy="1.25")

print("\n## Example 1")
# There are four resources to allocate among two people: Alice and George.
# The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice:
a = cvxpy.Variable(4)
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
LOGGER.setLevel(logging.INFO)
problem.solve()


print("\n## Example 3: leximax allocation\n")
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
