from cvxpy_leximin.objective import Leximax, Leximin
from cvxpy_leximin.problem import LOGGER, Problem
import pathlib

PARENT = pathlib.Path(__file__).parent.parent
__version__ = (PARENT / "VERSION").read_text().strip()
