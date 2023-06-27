from setuptools import setup

_VERSION = '0.0.0'

_REQUIRED_PACKAGES = [
    'torch==1.10.0',
    'torchvision==0.11.1',
    'fire==0.5.0',
    'fairscale==0.4.0',
    'Flask==2.0.2',
    'Flask-Cors==3.0.10',
    'pathlib==1.0.1',
    'requests==2.27.1',
    'numpy==1.19.5',
    'dataclasses==0.6',
    'sentencepiece==0.1.99'
]

_TEST_REQUIRES = [
    'bandit==1.7.4',
    'flake8==4.0.1',
    'pylint==2.6.2'
]

setup(
    name="llma",
    version=_VERSION.replace('-', ''),
    install_requires=_REQUIRED_PACKAGES,
    extras_require={'test': _TEST_REQUIRES}
)
