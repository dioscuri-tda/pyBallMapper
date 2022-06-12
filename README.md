# pyBallMapper

[![version](https://img.shields.io/badge/version-0.1-blue)](https://pypi.org/project/pyBallMapper)    

Python version of the Ball Mapper algorithm described in [arXiv:1901.07410 ](https://arxiv.org/abs/1901.07410) .  

### Installation  
```
pip install pyballmapper
```

### Basic usage
```
from pyballmapper import BallMapper
bm = BallMapper(points = my_pointcloud,    # the pointcloud, as a numpy array
                epsilon = 0.25)            # the radius of the covering balls
```

For more info check out the [example notebooks](https://github.com/dgurnari/pyBallMapper/tree/main/notebooks) .
