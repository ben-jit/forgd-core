from setuptools import setup, find_packages


setup(
    name='forgd_core',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'forgd_core = forgd_core.main:main',
        ],
    },
)
