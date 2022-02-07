import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent
print(f"\nHERE = {HERE.absolute()}\n")
README = (HERE / "README.md").read_text()
REQUIRES = (HERE / "requirements.txt").read_text().strip().split("\n")
REQUIRES = [lin.strip() for lin in REQUIRES]


setuptools.setup(
    name="cvxpy_leximin",
    # version is taken from setup.cfg, which takes it from cvxpy_leximin.__init__.py
    packages=setuptools.find_packages(),
    install_requires=REQUIRES,
    author="Erel Segal-Halevi",
    author_email="erelsgl@gmail.com",
    description="Let CVXPY support optimization in leximin order",
    keywords="optimization, cvxpy, leximin",
    license="Apache License, Version 2.0",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/erelsgl/cvxpy_leximin",
    project_urls={
        "Documentation": "https://github.com/erelsgl/cvxpy_leximin",
        "Bug Reports": "https://github.com/erelsgl/cvxpy_leximin/issues",
        "Source Code": "https://github.com/erelsgl/cvxpy_leximin",
    },
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
    ],
)

# Build:
#   [delete old folders: build, dist, *.egg_info, .venv_test]
#   python setup.py sdist bdist_wheel


# Publish to test PyPI:
#   twine upload --repository testpypi dist/*

# Publish to real PyPI:
#   twine upload --repository pypi dist/*

# Test in a virtual environment:
#    virtualenv .venv_test
#    .venv_test\Scripts\activate
#    pip install cvxpy_leximin
#    python examples\leximin_allocation.py
#    .venv\Scripts\activate
