from distutils.core import setup

setup(
    name='pyBallMapper',
    version='0.1.0',
    author='Davide Gurnari',
    author_email='davide.gurnari@gmail.com',
    packages=['pyballmapper', 'pyballmapper.tests'],
    url='https://github.com/dgurnari/pyBallMapper',
    license='LICENSE.txt',
    description='Python implementation of the Ball Mapper algorithm.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
        'matplotlib',
        'bokeh'
    ],
)
