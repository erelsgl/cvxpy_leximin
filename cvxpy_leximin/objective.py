"""
Define the Leximin and Leximax objectives.

Author: Erel Segal-Halevi.
Since:  2022-02.
"""

import abc
from typing import List

import cvxpy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.utilities import scopes


class MutliExpressionObjective(cvxpy.problems.objective.Objective):
    """An optimization objective that is made of multiple expressions.

    Parameters
    ----------
    expr: List[Expression]
        The expressions to act upon. Must be a scalars.

    Raises
    ------
    ValueError
        If one of the expressions is not a scalar or not real valued.
    """

    NAME = "multi expression objective"

    def __init__(self, expressions: List[Expression]) -> None:
        self.args = [Expression.cast_to_const(expr) for expr in expressions]
        # Validate that the objectives resolves to a scalar.
        for arg in self.args:
            if not arg.is_scalar():
                raise ValueError(
                    f"The '{self.NAME}' objective must resolve to a scalar."
                )
            if not arg.is_real():
                raise ValueError(
                    f"The '{self.NAME}' objective must be real valued."
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.args)})"

    def __str__(self) -> str:
        return " ".join([self.NAME, *[arg.name() for arg in self.args]])

    def __add__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __div__(self, other):
        raise NotImplementedError()

    @property
    def value(self) -> list:
        """The values of the objective expressions."""
        return [
            None if arg.value is None else scalar_value(arg.value)
            for arg in self.args
        ]

    def is_quadratic(self) -> bool:
        """Returns if all objectives are quadratic functions."""
        return all([arg.is_quadratic() for arg in self.args])

    def is_qpwa(self) -> bool:
        """Returns if all objectives are quadratic of piecewise affine."""
        return all([arg.is_qpwa() for arg in self.args])

    @abc.abstractmethod
    def is_arg_dcp(self, dpp: bool):
        pass

    def is_dcp(self, dpp: bool = False) -> bool:
        """All objectives must be convex/concave."""
        return all([self.is_arg_dcp(arg, dpp) for arg in self.args])

    @abc.abstractmethod
    def is_arg_dgp(self, dpp: bool):
        pass

    def is_dgp(self, dpp: bool = False) -> bool:
        """All objectives must be log-log convex/concave."""
        return all([self.is_log_log_convex(arg, dpp) for arg in self.args])

    def is_dpp(self, context="dcp") -> bool:
        with scopes.dpp_scope():
            if context.lower() == "dcp":
                return self.is_dcp(dpp=True)
            elif context.lower() == "dgp":
                return self.is_dgp(dpp=True)
            else:
                raise ValueError("Unsupported context ", context)


class Leximin(MutliExpressionObjective):
    """
    Parameters
    ----------
    expressions: Expression
        The list of expressions for which the leximin order should be applied.
        Must be scalars.

    Raises
    ------
    ValueError
        If one of the expressions is not a scalar or not real valued.
    """

    NAME = "leximin"

    def __neg__(self) -> "Leximax":
        """
        >>> x, y, z = cvxpy.Variable(3)
        >>> leximin_obj = Leximin([5*x + 3*y, 2*(1-x) + 4*(1-y) + 9*(1-z)])
        >>> leximin_obj
        Leximin([Expression(AFFINE, UNKNOWN, ()), Expression(AFFINE, UNKNOWN, ())])
        >>> -leximin_obj
        Leximax([Expression(AFFINE, UNKNOWN, ()), Expression(AFFINE, UNKNOWN, ())])
        """
        return Leximax([-arg for arg in self.args])

    def canonicalize(self) -> list:
        """Pass on the target expression's objective and constraints."""
        return [arg.canonical_form for arg in self.args]

    def is_arg_dcp(self, arg, dpp: bool) -> bool:
        """All arguments must be convex"""
        if dpp:
            with scopes.dpp_scope():
                return arg.is_convex()
        return arg.is_convex()

    def is_arg_dgp(self, arg, dpp: bool) -> bool:
        """All arguments must be log_log_convex"""
        if dpp:
            with scopes.dpp_scope():
                return arg.is_log_log_convex()
        return arg.is_log_log_convex()

    def is_dqcp(self) -> bool:
        """The objective must be quasiconvex."""
        return all([arg.is_quasiconvex() for arg in self.args])

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value."""
        return result


class Leximax(MutliExpressionObjective):
    """
    Parameters
    ----------
    expressions: Expression
        The list of expressions for which the leximax order should be applied. Must be scalars.

    Raises
    ------
    ValueError
        If one of the expressions is not a scalar or not real valued.
    """

    NAME = "leximax"

    def __neg__(self) -> Leximin:
        """
        >>> x, y, z = cvxpy.Variable(3)
        >>> leximax_obj = Leximax([5*x + 3*y, 2*(1-x) + 4*(1-y) + 9*(1-z)])
        >>> leximax_obj
        Leximax([Expression(AFFINE, UNKNOWN, ()), Expression(AFFINE, UNKNOWN, ())])
        >>> -leximax_obj
        Leximin([Expression(AFFINE, UNKNOWN, ()), Expression(AFFINE, UNKNOWN, ())])
        """
        return Leximin([-arg for arg in self.args])

    def canonicalize(self):
        """Negates the target expression's objective."""
        result = []
        for arg in self.args:
            obj, constraints = arg.canonical_form
            result.append((lu.neg_expr(obj), constraints))
        return result

    def is_arg_dcp(self, arg, dpp: bool) -> bool:
        """All arguments must be concave"""
        if dpp:
            with scopes.dpp_scope():
                return arg.is_concave()
        return arg.is_concave()

    def is_arg_dgp(self, arg, dpp: bool) -> bool:
        """All arguments must be log_log_convex"""
        if dpp:
            with scopes.dpp_scope():
                return arg.is_log_log_concave()
        return arg.is_log_log_concave()

    def is_dqcp(self) -> bool:
        """All objectives must be quasiconcave."""
        return all([arg.is_quasiconcave() for arg in self.args])

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value."""
        return -result


if __name__ == "__main__":
    import doctest

    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))
