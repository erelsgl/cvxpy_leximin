# cvxpy_leximin

Extends [cvxpy](https://github.com/cvxpy/cvxpy) by adding support for optimization in [leximin order](https://en.wikipedia.org/wiki/Leximin_order).

It can be used, for example, to find an egalitarian resource allocation:

    import cvxpy, logging
    from cvxpy_leximin import Problem, Leximin

    # There are four resources to allocate among two people: Alice and George.
    # The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice:
    a = cvxpy.Variable(4)
    # The following constraint represents the fact that the allocation is feasible:
    feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]

    # Alice values the resources at 5, 3, 0, 0:
    utility_Alice = a[0] * 5 + a[1] * 3 + a[2] * 0

    # George values the resources at 2, 4, 7, 0:
    utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9

    # A leximin allocation is an allocation that maximizes the sorted vector of utilities, in lexicographic order:
    objective = Leximin([utility_Alice, utility_George])

    # A problem is defined and solved like any cvxpy problem:
    problem = Problem(objective, constraints=feasible_allocation)
    problem.solve()
    print("Problem status: ", problem.status)   # optimal
    print("Objective value: ", objective.value)  
    # It is (8, 9). It maximizes the smallest utility (8), and subject to that, the next-smallest one (9).
    print("Allocation: ", a.value)
    # It is [1, 1, 0, 0]: Alice gets resources 0 and 1 (utility=8) and George resources 2 and 3 (utility=9).


For more examples, see the [examples folder](examples/).
