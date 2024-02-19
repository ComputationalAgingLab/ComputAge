from setuptools import setup

setup(
    name='computage',
    version='0.1',
    description='The library for using of aging clocks',
    packages=['computage'],
    install_requires=[
        'scikit-learn',
        'scipy',
        'statsmodels',
        'mapply',
        
    ],
    author_email='dmitrii.kriukov@skoltech.ru',
    zip_safe=False,
)
