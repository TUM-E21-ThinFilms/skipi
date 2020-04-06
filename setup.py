from setuptools import setup, find_packages

requires = ['numpy', 'scipy', 'matplotlib']

setup(
    name='dinv',
    version=__import__('skipi').__version__,
    author='Alexander Book',
    author_email='alexander.book@frm2.tum.de',
    license = 'MIT License',
    url='https://github.com/TUM-E21-ThinFilms/skipi',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
)
