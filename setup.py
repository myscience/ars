from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ars',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package requires
        *requirements
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts your package provides
        ],
    },
)