import pathlib
import setuptools

NAME = "cvxpy_leximin"
URL = "https://github.com/erelsgl/" + NAME
HERE = pathlib.Path(__file__).parent
print(f"\nHERE = {HERE.absolute()}\n")
README = (HERE / "README.md").read_text()
REQUIRES = (HERE / "requirements.txt").read_text().strip().split("\n")
REQUIRES = [lin.strip() for lin in REQUIRES]
print(f'\nVERSION = {(HERE / NAME / "VERSION").absolute()}\n')
VERSION = (HERE / NAME / "VERSION").read_text().strip()
# See https://packaging.python.org/en/latest/guides/single-sourcing-package-version/

setuptools.setup(
    name=NAME,
    packages=setuptools.find_packages(),
    version=VERSION,
    install_requires=REQUIRES,
    author="Erel Segal-Halevi",
    author_email="erelsgl@gmail.com",
    description="Let CVXPY support optimization in leximin order",
    keywords="optimization, cvxpy, leximin",
    license="Apache License, Version 2.0",
    license_files=("LICENSE",),
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={"Bug Reports": URL + "/issues", "Source Code": URL},
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
    ],
)

# Build:
#   Delete old folders: build, dist, *.egg_info, .venv_test.
#   Then run:
#        build
#   Or (old version):
#        python setup.py sdist bdist_wheel


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
