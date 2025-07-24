# CVXPY + Leximin

![Tox result](https://github.com/erelsgl/cvxpy_leximin/workflows/tox/badge.svg)
[![PyPI version](https://badge.fury.io/py/cvxpy-leximin.svg)](https://badge.fury.io/py/cvxpy-leximin)

The `cvxpy_leximin` package extends [cvxpy](https://github.com/cvxpy/cvxpy) by adding two objectives: `Leximin` and `Leximax`.
Each of these objectives takes as an argument a list of expressions.
Solving a problem with the `Leximin` objective follows the [leximin order](https://en.wikipedia.org/wiki/Leximin_order), that is:
 
* Find the solutions in which the smallest expression is as large as possible (subject to the constraints).
* If there are two or more such solutions, then among all these solutions, find the ones in which the next-smallest expression is as large as possible.
* If there are two or more such solutions, then among all these solutions, find the ones in which the third-smallest expression is as large as possible.
And so on.

The `Leximax` objective is solved in the opposite way: find the solutions that *minimize* the *largest* expression (subject to the constraints); among them,  minimize the next-largest expression; and so on.

Note that the current implementation works only when domain (as defined by the constraints) is convex. In particular, it does not work for integer programming.

## Installation

    pip install cvxpy_leximin

## Usage example

Leximin optimization can be used to find an egalitarian allocation of resources among people (see [Egalitarian item allocation](https://en.wikipedia.org/wiki/Egalitarian_item_allocation).)

    import cvxpy, logging
    from cvxpy_leximin import Problem, Leximin

    # There are four resources to allocate among two people: Alice and George.
    # The variables a[0], a[1], a[2], a[3] denote the fraction of each resource given to Alice:
    a = cvxpy.Variable(4)

    # The following constraint represents the fact that the allocation is feasible:
    feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]

    # Alice values the resources at 5, 3, 0, 0:
    utility_Alice = a[0] * 5 + a[1] * 3 + a[2] * 0

    # George values the resources at 2, 4, 9, 0:
    utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9

    # The leximin objective is: maximize the smallest utility, and subject to that, the next-smallest utility.
    objective = Leximin([utility_Alice, utility_George])

    # A problem is defined and solved like any cvxpy problem:
    problem = Problem(objective, constraints=feasible_allocation)
    problem.solve()
    print("Problem status: ", problem.status)   # Should be optimal
    print("Objective value: ", objective.value)  
       # It is (8, 9). It maximizes the smallest utility (8), and subject to that, the next-smallest one (9).
    print("Allocation: ", a.value)
       # It is [1, 1, 0, 0]: Alice gets resources 0 and 1 (utility=8) and George resources 2 and 3 (utility=9).


For more examples, see the [examples folder](examples/).

## Credits

The ordered outcomes and ordered values algorithms are based on 
> Ogryczak and Sliwinski (2006),
> ["On Direct Methods for Lexicographic Min-Max Optimization"](https://link.springer.com/chapter/10.1007/11751595_85).
Programmed by Moshe Ofer.

The saturation algorithm is based on:
> [Stephen J. Willson](https://faculty.sites.iastate.edu/swillson/),
> "Fair Division Using Linear Programming" (1998).
> Part 6, pages 20--27.
I am grateful to Sylvain Bouveret for his help with the algorithm. All remaining errors and bugs are my own.


## Status

The functionality was tested only on fair allocation problems, only on objectives with linear expressions, and only on the default solver (SCIPY).

If you would like to contribute, it could be great to test leximin optimization on other kinds of problems, expressions and solvers.


