# Getting started

[![version](https://img.shields.io/badge/version-0.2-blue)](https://pypi.org/project/pyBallMapper)    

Python version of the Ball Mapper algorithm described in [arXiv:1901.07410 ](https://arxiv.org/abs/1901.07410) .  

## Install the package 📦
```
pip install pyballmapper
```

## Basic usage
```
from pyballmapper import BallMapper
bm = BallMapper(X = my_pointcloud,    # the pointcloud, as a numpy array
                eps = 0.25)           # the radius of the covering balls

bm.draw_networkx()
```

For more info check out the [example notebooks](https://github.com/dgurnari/pyBallMapper/tree/main/notebooks) .


