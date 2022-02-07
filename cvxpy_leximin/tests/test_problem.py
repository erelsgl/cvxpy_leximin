"""
Test that the new Problem class handles the Leximin objective.
"""

from cvxpy import Variable
from cvxpy.constraints import NonPos, Zero
from cvxpy.tests.base_test import BaseTest

from cvxpy_leximin import Leximax, Leximin, Problem


class TestProblem(BaseTest):
    def setUp(self) -> None:
        self.a = Variable(name="a")
        self.b = Variable(name="b")
        self.c = Variable(name="c")

        self.x = Variable(2, name="x")
        self.y = Variable(3, name="y")
        self.z = Variable(2, name="z")

        self.A = Variable((2, 2), name="A")
        self.B = Variable((2, 2), name="B")
        self.C = Variable((3, 2), name="C")

    def test_to_str(self) -> None:
        """Test string representations."""
        obj = Leximin([self.a, self.b, self.c])
        prob = Problem(obj)
        self.assertEqual(repr(prob), f"Problem({repr(obj)}, {repr([])})")
        constraints = [self.x * 2 == self.x, self.x == 0]
        prob = Problem(obj, constraints)
        self.assertEqual(
            repr(prob), f"Problem({repr(obj)}, {repr(constraints)})"
        )

        # Test str.
        result = "leximin a b c\nsubject to a == 0\n           b <= 0"
        prob = Problem(
            Leximin([self.a, self.b, self.c]), [Zero(self.a), NonPos(self.b)]
        )
        self.assertEqual(str(prob), result)

    def test_leximin_1(self) -> None:
        a = Variable(4)
        feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]
        utility_Alice = a[0] * 5 + a[1] * 3 + a[2] * 0
        utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9
        objective = Leximin([utility_Alice, utility_George])
        problem = Problem(objective, constraints=feasible_allocation)
        problem.solve()
        self.assertEqual(problem.status, "optimal")
        self.assertAlmostEqual(utility_Alice.value, 8, places=3)
        self.assertAlmostEqual(utility_George.value, 9, places=3)
        self.assertAlmostEqual(objective.value, [8, 9], places=3)
        self.assertAlmostEqual(problem.value, [8, 9], places=3)
        self.assertAlmostEqual(list(a.value), [1, 1, 0, 0], places=3)

    def test_leximin_2(self) -> None:
        a = Variable(4)
        feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]
        utility_Alice = -5 * a[0] - 3 * a[1] - 0 * a[2]
        utility_George = -2 * (1 - a[0]) - 4 * (1 - a[1]) - 9 * (1 - a[2])
        problem = Problem(
            Leximin([utility_Alice, utility_George]),
            constraints=feasible_allocation,
        )
        problem.solve()
        self.assertEqual(problem.status, "optimal")
        self.assertAlmostEqual(utility_Alice.value, -2.57, places=2)
        self.assertAlmostEqual(utility_George.value, -2.57, places=2)
        self.assertAlmostEqual(a[0].value, 0, places=2)
        self.assertAlmostEqual(a[1].value, 0.857, places=2)
        self.assertAlmostEqual(a[2].value, 1, places=2)
        self.assertAlmostEqual(a[3].value, 0, places=2)

    def test_leximin_3(self) -> None:
        a = Variable(4)
        feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]
        utility_Alice = (
            (1 / 3) * a[0] + 0 * a[1] + (1 / 3) * a[2] + (1 / 3) * a[3]
        )
        utility_George = (
            1 * (1 - a[0]) + 1 * (1 - a[1]) + 1 * (1 - a[2]) + 0 * (1 - a[3])
        )
        problem = Problem(
            Leximin([utility_Alice, utility_George]),
            constraints=feasible_allocation,
        )
        problem.solve()
        self.assertEqual(problem.status, "optimal")
        self.assertAlmostEqual(utility_Alice.value, 1, places=2)
        self.assertAlmostEqual(utility_George.value, 1, places=2)
        self.assertAlmostEqual(a[0].value, 1, places=2)
        self.assertAlmostEqual(a[1].value, 0, places=2)
        self.assertAlmostEqual(a[2].value, 1, places=2)
        self.assertAlmostEqual(a[3].value, 1, places=2)

    def test_leximax_1(self) -> None:
        a = Variable(4)
        feasible_allocation = [x >= 0 for x in a] + [x <= 1 for x in a]
        utility_Alice = a[0] * 5 + a[1] * 2 + a[2] * 0
        utility_George = (1 - a[0]) * 2 + (1 - a[1]) * 4 + (1 - a[2]) * 9
        problem = Problem(
            Leximax([utility_Alice, utility_George]),
            constraints=feasible_allocation,
        )
        problem.solve()
        self.assertEqual(problem.status, "optimal")
        self.assertAlmostEqual(utility_Alice.value, 2, places=2)
        self.assertAlmostEqual(utility_George.value, 2, places=2)
        self.assertAlmostEqual(a[0].value, 0, places=2)
        self.assertAlmostEqual(a[1].value, 1, places=2)
        self.assertAlmostEqual(a[2].value, 1, places=2)
        self.assertAlmostEqual(a[3].value, 0, places=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
