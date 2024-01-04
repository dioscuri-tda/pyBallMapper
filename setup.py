from distutils.core import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="pyBallMapper",
    version="0.3.3",
    author="Davide Gurnari",
    author_email="davide.gurnari@gmail.com",
    packages=["pyballmapper", "pyballmapper.tests"],
    url="https://github.com/dgurnari/pyBallMapper",
    license="LICENSE.txt",
    description="Python implementation of the Ball Mapper algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "matplotlib",
        "bokeh",
        "numba",
        "scikit-learn",
    ],
)
