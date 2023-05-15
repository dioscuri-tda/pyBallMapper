from distutils.core import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='pyBallMapper',
    version='0.2.2',
    author='Davide Gurnari',
    author_email='davide.gurnari@gmail.com',
    packages=['pyballmapper', 'pyballmapper.tests'],
    url='https://github.com/dgurnari/pyBallMapper',
    license='LICENSE.txt',
    description='Python implementation of the Ball Mapper algorithm.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
        'matplotlib',
        'bokeh',
        'numba',
        'scikit-learn'
    ],
)
