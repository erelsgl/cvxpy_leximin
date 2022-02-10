from cvxpy_leximin.objective import Leximax, Leximin
from cvxpy_leximin.problem import LOGGER, Problem
import pathlib

HERE = pathlib.Path(__file__).parent
__version__ = (HERE / "VERSION").read_text().strip()
