import setuptools, pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
REQUIRES = (HERE / "requirements.txt").read_text().strip().split("\n")
REQUIRES = [lin.strip() for lin in REQUIRES]

# from cvxpy_leximin import __version__
__version__ = "0.1.0"

setuptools.setup(
    name="cvxpy_leximin",
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=REQUIRES,
    author="Erel Segal-Halevi",
    author_email="erelsgl@gmail.com",
    description="Let CVXPY support optimization in leximin order",
    keywords="optimization",
    license="Apache License, Version 2.0",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/erelsgl/cvxpy_leximin",
    project_urls={
        "Documentation": "https://github.com/erelsgl/cvxpy_leximin",
        "Bug Reports": "https://github.com/erelsgl/cvxpy_leximin/issues",
        "Source Code": "https://github.com/erelsgl/cvxpy_leximin",
    },
    python_requires=">=3.9",
    include_package_data=True,
)

# Build:
#   [delete old folders: build, dist, test_env]
#   python setup.py sdist bdist_wheel


# Publish to test PyPI:
#   twine upload --repository testpypi dist/*

# Publish to real PyPI:
#   twine upload --repository pypi dist/*


# Test in a virtual environment:
#    cd ..
#    virtualenv test_env
#    test_env\Scripts\activate
#    pip install -i https://test.pypi.org/simple/ cvxpy_leximin
#    pytest test_env\Lib\site-packages\cvxpy_leximin
