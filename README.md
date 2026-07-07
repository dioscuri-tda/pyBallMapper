# pyBallMapper

[![PyPI version](https://img.shields.io/pypi/v/pyBallMapper.svg?color=blue)](https://pypi.org/project/pyBallMapper)
[![Documentation Status](https://readthedocs.org/projects/pyballmapper/badge/?version=latest)](https://pyballmapper.readthedocs.io/en/latest/?badge=latest)

Python version of the BallMapper algorithm described in [arXiv:1901.07410 ](https://arxiv.org/abs/1901.07410) .  

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/jones_17_bm_white.png">
  <img alt="Jones 17" src="img/jones_17_bm.png">
</picture>

### Install the package 📦   

```
pip install pyballmapper
```

For development, use [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/dioscuri-tda/pyBallMapper.git
cd pyBallMapper
uv sync                    # install dependencies + project
uv run --group dev pre-commit install   # install pre-commit & enable git hooks
uv run --group docs ...    # when building docs
```

### Basic usage
```
from pyballmapper import BallMapper
bm = BallMapper(X = my_pointcloud,    # the pointcloud, as a array-like of shape (n_samples, n_features)
                eps = 4.669)          # the radius of the covering balls
```

For more info check out the [example notebooks](https://github.com/dgurnari/pyBallMapper/tree/main/notebooks) or the [documentation](https://pyballmapper.readthedocs.io).
