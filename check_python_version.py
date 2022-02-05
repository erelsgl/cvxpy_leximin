import sys

# print("\nPython version: ", sys.version)
# print()

print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)

from pip import _internal

_internal.main(["list"])

# import cvxpy


# class Test(cvxpy.utilities.Canonical):
#     pass
